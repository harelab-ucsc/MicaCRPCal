"""
AutoCalNode — automatic exposure lock and irradiance calibration at 3 m AGL.

Waits until the drone clears ALT_THRESHOLD_M (radalt), then:

  1. Waits for libcamera's built-in AE/AGC to converge: CONVERGE_FRAMES
     consecutive frames where the 99th-pct DN of the representative band
     stays below BRIGHT_CEIL and the 5th-pct stays above DARK_FLOOR.

  2. Reads back the settled ExposureTime and AnalogueGain from the camera
     node's ae_exposure_time_us / ae_analogue_gain parameters, which the
     camera_ros driver populates each frame from libcamera request metadata.

  3. Locks both cameras at those settings (AeEnable=False + explicit values).

  4. Snapshots the AS7265x irradiance and publishes it on /panel_cal/spec_ref
     (latched) so stream_processor can apply per-cycle irradiance correction.

  5. Exits.

cam0 (multispectral, 4 bands side-by-side) and cam1 (Bayer RGB, off-nadir)
are calibrated sequentially — their lenses differ so they need independent
exposures.
"""

import threading
import time

import numpy as np
import rclpy
import rclpy.executors
from cv_bridge import CvBridge
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray

try:
    from custom_msgs.msg import AltSNR
except ImportError:
    AltSNR = None

try:
    from as7265x_at_msgs.msg import AS7265xCal
except ImportError:
    AS7265xCal = None

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

ALT_THRESHOLD_M = 3.0

# Brightness targets (as fraction of dtype_max).
BRIGHT_CEIL = 0.85   # 99th-pct of representative band must stay below this
DARK_FLOOR  = 0.05   # 5th-pct of representative band must stay above this

# Require this many consecutive in-bounds frames before declaring convergence.
CONVERGE_FRAMES = 5

# Lock at current AE state if convergence hasn't been reached within this time.
CONVERGE_TIMEOUT_S = 20.0

NUM_CAM0_SLICES = 4

_LATCHED_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    depth=1,
)
_SNS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _analyze_cam0(img: np.ndarray) -> tuple[float, float]:
    """Return (bright_99, dark_05) for the 735 nm band (slice 2).

    Calibrating against the full-band max/min fails indoors because 450 nm
    and 850 nm bands have near-zero signal while 695 nm saturates, making the
    constraints contradictory. The 735 nm band (slice 2) is the 2nd-least-dark
    band and gives a representative exposure target for flight.
    """
    w = img.shape[1]
    sw = w // NUM_CAM0_SLICES
    sl = img[:, 2 * sw: 3 * sw].astype(np.float64)
    return float(np.percentile(sl, 99)), float(np.percentile(sl, 5))


def _analyze_cam1(img: np.ndarray) -> tuple[float, float]:
    """Return (bright_99, dark_05) as worst-case across the R, G, B Bayer channels.

    cam1 uses SBGGR16 (Sony Bayer, Blue-Green-Green-Red quad):
      B  at even row, even col
      Gr at even row, odd  col
      Gb at odd  row, even col
      R  at odd  row, odd  col

    Analysing channels separately catches a clipped R or B that would be masked
    by the dominant (2×) green population in a whole-frame percentile.
    """
    arr = img.astype(np.float64)
    channels = [
        arr[0::2, 0::2].ravel(),                                   # B
        np.concatenate([arr[0::2, 1::2].ravel(),
                        arr[1::2, 0::2].ravel()]),                 # G (Gr + Gb)
        arr[1::2, 1::2].ravel(),                                   # R
    ]
    p99s = [float(np.percentile(ch, 99)) for ch in channels]
    p05s = [float(np.percentile(ch, 5))  for ch in channels]
    return max(p99s), min(p05s)


def _param(name: str, value) -> Parameter:
    if isinstance(value, bool):
        return Parameter(
            name=name,
            value=ParameterValue(
                type=ParameterType.PARAMETER_BOOL, bool_value=value
            ),
        )
    if isinstance(value, int):
        return Parameter(
            name=name,
            value=ParameterValue(
                type=ParameterType.PARAMETER_INTEGER, integer_value=value
            ),
        )
    return Parameter(
        name=name,
        value=ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE, double_value=float(value)
        ),
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class AutoCalNode(Node):

    def __init__(self):
        super().__init__("auto_cal")
        self._bridge = CvBridge()

        self.declare_parameter("force_cal", False)
        self._force_cal: bool = self.get_parameter("force_cal").value

        self._above_alt = threading.Event()
        self._alt_notified = False
        if self._force_cal:
            self._above_alt.set()
            self.get_logger().warn(
                "force_cal=True — skipping altitude gate and running "
                "auto-calibration immediately. Only use this on the ground "
                "for testing."
            )

        # Per-camera frame buffers and notification events
        self._cam0_lock = threading.Lock()
        self._cam0_frame: np.ndarray | None = None
        self._cam0_evt = threading.Event()

        self._cam1_lock = threading.Lock()
        self._cam1_frame: np.ndarray | None = None
        self._cam1_evt = threading.Event()

        # Spectrometer buffer
        self._spec_lock = threading.Lock()
        self._latest_spec: list | None = None

        self._pub_spec_ref = self.create_publisher(
            Float32MultiArray, "/panel_cal/spec_ref", _LATCHED_QOS
        )
        # /cal/exposure_locked — latched Bool published once cameras are locked.
        # panel_scan subscribes and only opens its QR-scan window after receiving
        # it, ensuring the panel is imaged at the same exposure as the flight.
        self._pub_exposure_locked = self.create_publisher(
            Bool, "/cal/exposure_locked", _LATCHED_QOS
        )

        # Service clients for each camera.
        # NOTE: not named _clients — rclpy.Node uses self._clients internally
        # to track all service clients; shadowing it breaks the executor.
        self._set_param_clients = {
            "cam0": self.create_client(
                SetParameters, "/cam0/camera_node/set_parameters"
            ),
            "cam1": self.create_client(
                SetParameters, "/cam1/camera_node/set_parameters"
            ),
        }
        self._get_param_clients = {
            "cam0": self.create_client(
                GetParameters, "/cam0/camera_node/get_parameters"
            ),
            "cam1": self.create_client(
                GetParameters, "/cam1/camera_node/get_parameters"
            ),
        }

        # Subscriptions
        if AltSNR is not None:
            self.create_subscription(
                AltSNR, "/rad_altitude", self._radalt_cb, _SNS_QOS
            )
        else:
            self.get_logger().warn(
                "custom_msgs not available — radalt sub disabled; "
                "AutoCalNode will never trigger."
            )

        self.create_subscription(
            Image,
            "/cam0/camera_node/image_raw",
            self._cam0_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            "/cam1/camera_node/image_raw",
            self._cam1_cb,
            qos_profile_sensor_data,
        )

        if AS7265xCal is not None:
            self.create_subscription(
                AS7265xCal,
                "/as7265x/calibrated_values",
                self._spec_cb,
                _SNS_QOS,
            )
        else:
            self.get_logger().warn(
                "as7265x_at_msgs not available — "
                "/panel_cal/spec_ref will not be published."
            )

        # Run the calibration sequence in a background thread so blocking
        # waits don't starve the executor (which runs in the main thread via
        # MultiThreadedExecutor).
        threading.Thread(target=self._cal_task, daemon=True).start()

        if self._force_cal:
            self.get_logger().info(
                f"AutoCalNode ready — force_cal active, starting immediately. "
                f"Convergence: {CONVERGE_FRAMES} stable frames in "
                f"[{DARK_FLOOR*100:.0f}%, {BRIGHT_CEIL*100:.0f}%]"
            )
        else:
            self.get_logger().info(
                f"AutoCalNode ready — waiting for {ALT_THRESHOLD_M} m AGL. "
                f"Convergence: {CONVERGE_FRAMES} stable frames in "
                f"[{DARK_FLOOR*100:.0f}%, {BRIGHT_CEIL*100:.0f}%]"
            )

    # -----------------------------------------------------------------------
    # ROS callbacks
    # -----------------------------------------------------------------------

    def _radalt_cb(self, msg) -> None:
        if msg.altitude > ALT_THRESHOLD_M:
            if not self._alt_notified:
                self._alt_notified = True
                self.get_logger().info(
                    f"[AUTO-CAL] altitude gate cleared ({msg.altitude:.1f} m AGL >= "
                    f"{ALT_THRESHOLD_M} m threshold) — triggering calibration"
                )
            self._above_alt.set()

    def _cam0_cb(self, msg: Image) -> None:
        try:
            raw = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception:
            return
        with self._cam0_lock:
            self._cam0_frame = raw
        self._cam0_evt.set()

    def _cam1_cb(self, msg: Image) -> None:
        try:
            raw = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception:
            return
        with self._cam1_lock:
            self._cam1_frame = raw
        self._cam1_evt.set()

    def _spec_cb(self, msg) -> None:
        with self._spec_lock:
            self._latest_spec = list(msg.values)

    # -----------------------------------------------------------------------
    # Parameter get / set
    # -----------------------------------------------------------------------

    def _set_params(self, cam: str, params: list, timeout: float = 5.0) -> bool:
        """Submit a SetParameters call and poll for completion."""
        client = self._set_param_clients[cam]
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                f"{cam}: set_parameters service unavailable"
            )
            return False
        req = SetParameters.Request()
        req.parameters = params
        future = client.call_async(req)
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > deadline:
                self.get_logger().error(f"{cam}: set_parameters timed out")
                return False
            time.sleep(0.01)
        result = future.result()
        if result is None:
            return False
        rejected = [
            (p.name, r.reason)
            for p, r in zip(params, result.results)
            if not r.successful
        ]
        if not rejected:
            return True

        # The libcamera Raspberry Pi IPA returns successful=False for ExposureTime
        # when ExposureTimeMode is also active, but the value IS applied anyway.
        SPURIOUS = "ExposureTimeMode and ExposureTime must not be set simultaneously"
        spurious = [(n, r) for n, r in rejected if SPURIOUS in r]
        real = [(n, r) for n, r in rejected if SPURIOUS not in r]

        if spurious:
            self.get_logger().debug(
                f"{cam}: ExposureTime returned successful=False due to "
                f"ExposureTimeMode conflict, but value is applied — ignoring."
            )
        if real:
            sep = "=" * 62
            details = "".join(f"    {name}: {reason}\n" for name, reason in real)
            self.get_logger().error(
                f"\n{sep}\n"
                f"  CAMERA PARAMETER UPDATE REJECTED — {cam}\n"
                f"  The camera driver refused the following settings:\n"
                f"{details}"
                f"  The parameter was NOT applied.\n"
                f"{sep}"
            )
            return False
        return True

    def _get_ae_params(self, cam: str, timeout: float = 5.0) -> tuple[int, float]:
        """Read ae_exposure_time_us and ae_analogue_gain from the camera node.

        These parameters are updated each frame by camera_ros from libcamera
        request metadata and reflect the values the IPA actually applied.
        """
        client = self._get_param_clients[cam]
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f"{cam}: get_parameters service unavailable")
            return 5000, 1.0
        req = GetParameters.Request()
        req.names = ["ae_exposure_time_us", "ae_analogue_gain"]
        future = client.call_async(req)
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > deadline:
                self.get_logger().error(f"{cam}: get_parameters timed out")
                return 5000, 1.0
            time.sleep(0.01)
        result = future.result()
        if result is None or len(result.values) < 2:
            self.get_logger().error(
                f"{cam}: get_parameters returned unexpected result"
            )
            return 5000, 1.0
        exp_us = int(result.values[0].integer_value)
        gain = float(result.values[1].double_value)
        return exp_us, gain

    # -----------------------------------------------------------------------
    # Frame acquisition (blocks the calling background thread)
    # -----------------------------------------------------------------------

    def _next_frame(self, cam: str, timeout: float = 2.0) -> np.ndarray | None:
        evt = self._cam0_evt if cam == "cam0" else self._cam1_evt
        lock = self._cam0_lock if cam == "cam0" else self._cam1_lock
        buf_attr = "_cam0_frame" if cam == "cam0" else "_cam1_frame"
        evt.clear()
        if not evt.wait(timeout):
            self.get_logger().warn(f"Timeout waiting for {cam} frame")
            return None
        with lock:
            return getattr(self, buf_attr).copy()

    # -----------------------------------------------------------------------
    # AE convergence wait
    # -----------------------------------------------------------------------

    def _wait_for_ae(self, cam: str) -> tuple[int, float]:
        """Wait for libcamera AE to settle, then read back and return
        (exposure_us, gain).

        Declares convergence once CONVERGE_FRAMES consecutive frames have
        bright_99 < BRIGHT_CEIL and dark_05 > DARK_FLOOR. If convergence
        doesn't happen within CONVERGE_TIMEOUT_S, locks at the current state.
        """
        analyze = _analyze_cam0 if cam == "cam0" else _analyze_cam1
        dtype_max = 65535.0
        bright_thresh = BRIGHT_CEIL * dtype_max
        dark_thresh = DARK_FLOOR * dtype_max

        consecutive = 0
        deadline = time.monotonic() + CONVERGE_TIMEOUT_S

        self.get_logger().info(
            f"{cam}: waiting for AE to converge "
            f"({CONVERGE_FRAMES} consecutive frames in "
            f"[{DARK_FLOOR*100:.0f}%, {BRIGHT_CEIL*100:.0f}%])"
        )

        while time.monotonic() < deadline:
            frame = self._next_frame(cam)
            if frame is None:
                continue

            bright_99, dark_05 = analyze(frame)
            self.get_logger().debug(
                f"{cam}: bright_99={bright_99/dtype_max*100:.1f}%  "
                f"dark_05={dark_05/dtype_max*100:.1f}%  "
                f"stable={consecutive}/{CONVERGE_FRAMES}"
            )

            if bright_99 < bright_thresh and dark_05 > dark_thresh:
                consecutive += 1
                if consecutive >= CONVERGE_FRAMES:
                    self.get_logger().info(f"{cam}: AE converged")
                    break
            else:
                if consecutive > 0:
                    self.get_logger().debug(
                        f"{cam}: stability reset — "
                        f"bright_99={bright_99/dtype_max*100:.1f}%  "
                        f"dark_05={dark_05/dtype_max*100:.1f}%"
                    )
                consecutive = 0
        else:
            self.get_logger().warn(
                f"{cam}: AE did not converge within {CONVERGE_TIMEOUT_S:.0f}s — "
                "locking at current state"
            )

        exp_us, gain = self._get_ae_params(cam)
        self.get_logger().info(
            f"{cam}: AE settled at ExposureTime={exp_us} µs  AnalogueGain={gain:.2f}×"
        )
        return exp_us, gain

    # -----------------------------------------------------------------------
    # Main calibration sequence (background thread)
    # -----------------------------------------------------------------------

    def _cal_task(self) -> None:
        if not self._force_cal:
            self.get_logger().info(
                f"auto_cal: waiting for drone to clear {ALT_THRESHOLD_M} m AGL "
                "before locking exposure and capturing irradiance reference."
            )
        while not self._above_alt.wait(timeout=5.0):
            self.get_logger().warn(
                f"auto_cal: still waiting for {ALT_THRESHOLD_M} m AGL — "
                "cameras are in auto-exposure mode, images are not yet being saved."
            )
        sep = "=" * 62
        self.get_logger().info(
            f"\n{sep}\n"
            f"  AUTO-CALIBRATION STARTING\n"
            f"  Drone cleared {ALT_THRESHOLD_M} m AGL — locking cam0 + cam1 exposure\n"
            f"  Images will NOT be saved until this completes.\n"
            f"{sep}"
        )

        for cam in ("cam0", "cam1"):
            exp_us, gain = self._wait_for_ae(cam)
            self._set_params(cam, [
                _param("AeEnable", False),
                _param("ExposureTime", exp_us),
                _param("AnalogueGain", gain),
            ])
            self.get_logger().info(
                f"{cam} locked — ExposureTime={exp_us} µs  AnalogueGain={gain:.2f}×"
            )

        with self._spec_lock:
            spec = self._latest_spec

        if spec is not None:
            msg = Float32MultiArray()
            msg.data = [float(v) for v in spec]
            self._pub_spec_ref.publish(msg)
            self.get_logger().info(
                f"Irradiance reference published ({len(spec)} bands) "
                "on /panel_cal/spec_ref"
            )
        else:
            self.get_logger().warn(
                "No spectrometer data — /panel_cal/spec_ref not published. "
                "Per-cycle irradiance correction will be skipped."
            )

        self._pub_exposure_locked.publish(Bool(data=True))
        sep = "=" * 62
        self.get_logger().info(
            f"\n{sep}\n"
            f"  AUTO-CALIBRATION COMPLETE\n"
            f"  Both cameras locked at flight exposure.\n"
            f"  /panel_cal/spec_ref published — stream_processor will now\n"
            f"  save and spectral-correct images.\n"
            f"  panel_scan QR window is now open.\n"
            f"{sep}"
        )
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    executor = rclpy.executors.MultiThreadedExecutor()
    node = AutoCalNode()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
