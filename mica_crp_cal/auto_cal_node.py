"""
AutoCalNode — automatic exposure and irradiance calibration at 6 m AGL.

Waits until the drone clears ALT_THRESHOLD_M (radalt), then:

  1. Binary-searches ExposureTime and AnalogueGain for cam0 and cam1 so that:
       - 99th-percentile DN of the BRIGHTEST band stays below BRIGHT_CEIL.
       - 5th-percentile DN of the DARKEST band stays above DARK_FLOOR.
     Gain is increased only when the maximum allowed exposure is insufficient
     to lift the darkest band above the floor. This keeps noise as low as
     possible while avoiding motion blur.

  2. Locks both cameras at the found settings (AeEnable=False).

  3. Snapshots the AS7265x irradiance and publishes it on /panel_cal/spec_ref
     (latched) so stream_processor can apply the per-cycle irradiance ratio
     correction throughout the flight.

  4. Exits.

Conservative limits (tune in constants below):
  MAX_EXPOSURE_US = 5 000 µs (5 ms)  — < 1 px smear at 5 m/s, 0.03 m/px GSD
  MAX_GAIN        = 6.0              — beyond this, analogue noise degrades
                                       reflectance accuracy on small sensors

cam0 (multispectral, 4 bands side-by-side on one sensor) and cam1 (Bayer RGB,
off-nadir) have independent exposures because they are separate cameras with
different lenses. cam0 analysis uses per-slice min/max; cam1 uses the full frame.
"""

import threading
import time

import numpy as np
import rclpy
import rclpy.executors
from cv_bridge import CvBridge
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
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
# Tunable limits
# ---------------------------------------------------------------------------

ALT_THRESHOLD_M = 6.0

# Exposure: 5 ms gives < 1 px motion smear at 5 m/s with 0.03 m/px GSD.
MAX_EXPOSURE_US = 5_000
MIN_EXPOSURE_US = 100

# Gain: try these levels in order, stepping up only when MAX_EXPOSURE is not
# enough to satisfy DARK_FLOOR. Higher gain → more analogue noise.
GAIN_STEPS = (1.0, 2.0, 4.0, 6.0)

# Brightness targets (as fraction of dtype_max).
BRIGHT_CEIL = 0.85   # 99th-pct of brightest band must stay below this
DARK_FLOOR = 0.05    # 5th-pct of darkest band must stay above this

# Binary-search budget per gain level.
MAX_ITERS = 8

# How many frames to skip after a parameter change before sampling.
# At 3 Hz one frame = 333 ms; skipping 1 gives ~333 ms settling time.
SETTLE_FRAMES = 1

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

NUM_CAM0_SLICES = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _analyze_cam0(img: np.ndarray) -> tuple[float, float]:
    """Return (bright_99, dark_05) across all four band slices."""
    w = img.shape[1]
    sw = w // NUM_CAM0_SLICES
    arr = img.astype(np.float64)
    p99s, p05s = [], []
    for i in range(NUM_CAM0_SLICES):
        sl = arr[:, i * sw: (i + 1) * sw]
        p99s.append(float(np.percentile(sl, 99)))
        p05s.append(float(np.percentile(sl, 5)))
    return max(p99s), min(p05s)


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

        # Altitude gate — set immediately if force_cal is True.
        self.declare_parameter("force_cal", False)
        self._force_cal: bool = self.get_parameter("force_cal").value
        self._above_alt = threading.Event()
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

        # /panel_cal/spec_ref publisher
        self._pub_spec_ref = self.create_publisher(
            Float32MultiArray, "/panel_cal/spec_ref", _LATCHED_QOS
        )
        # /cal/exposure_locked — latched Bool published once cameras are locked.
        # panel_scan subscribes to this and only starts its QR-scan window after
        # receiving it, ensuring the panel is imaged at the same exposure the
        # flight images will use.
        self._pub_exposure_locked = self.create_publisher(
            Bool, "/cal/exposure_locked", _LATCHED_QOS
        )

        # SetParameters service clients for each camera
        self._clients = {
            "cam0": self.create_client(
                SetParameters, "/cam0/camera_node/set_parameters"
            ),
            "cam1": self.create_client(
                SetParameters, "/cam1/camera_node/set_parameters"
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
                f"Exposure limit: {MAX_EXPOSURE_US} µs | Gain limit: {max(GAIN_STEPS):.1f}×"
            )
        else:
            self.get_logger().info(
                f"AutoCalNode ready — waiting for {ALT_THRESHOLD_M} m AGL. "
                f"Exposure limit: {MAX_EXPOSURE_US} µs | Gain limit: {max(GAIN_STEPS):.1f}×"
            )

    # -----------------------------------------------------------------------
    # ROS callbacks
    # -----------------------------------------------------------------------

    def _radalt_cb(self, msg) -> None:
        if msg.altitude > ALT_THRESHOLD_M:
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
    # Parameter setting
    # -----------------------------------------------------------------------

    def _set_params(self, cam: str, params: list, timeout: float = 5.0) -> bool:
        """Submit a SetParameters call and poll for completion.

        The executor (MultiThreadedExecutor, main thread) will process the
        service response and resolve the future; this thread polls until done.
        """
        client = self._clients[cam]
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
        return future.result() is not None

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

    def _settled_frame(self, cam: str) -> np.ndarray | None:
        """Discard SETTLE_FRAMES then return the next frame."""
        for _ in range(SETTLE_FRAMES):
            if self._next_frame(cam) is None:
                return None
        return self._next_frame(cam)

    # -----------------------------------------------------------------------
    # Exposure binary search
    # -----------------------------------------------------------------------

    def _calibrate(self, cam: str) -> tuple[int, float]:
        """
        Binary-search ExposureTime at increasing gain levels until both
        BRIGHT_CEIL and DARK_FLOOR constraints are satisfied.

        Returns (locked_exposure_us, locked_gain).
        """
        analyze = _analyze_cam0 if cam == "cam0" else _analyze_cam1
        dtype_max = 65535.0  # both cameras configured for 16-bit output

        bright_thresh = BRIGHT_CEIL * dtype_max
        dark_thresh = DARK_FLOOR * dtype_max

        final_exposure = MAX_EXPOSURE_US
        final_gain = GAIN_STEPS[-1]

        for gain in GAIN_STEPS:
            lo, hi = MIN_EXPOSURE_US, MAX_EXPOSURE_US
            exposure = (lo + hi) // 2

            self.get_logger().info(
                f"{cam}: starting binary search — "
                f"gain={gain:.1f}  exposure range [{lo}, {hi}] µs"
            )

            # Disable AE and seed the binary search.
            self._set_params(cam, [
                _param("AeEnable", False),
                _param("ExposureTime", exposure),
                _param("AnalogueGain", gain),
            ])

            bright_99 = dark_05 = 0.0

            for it in range(MAX_ITERS):
                frame = self._settled_frame(cam)
                if frame is None:
                    self.get_logger().error(
                        f"{cam}: no frame — aborting calibration, "
                        f"locking at current settings"
                    )
                    return exposure, gain

                bright_99, dark_05 = analyze(frame)
                self.get_logger().info(
                    f"{cam} iter {it}: exposure={exposure} µs  gain={gain:.1f}  "
                    f"bright_99={bright_99 / dtype_max * 100:.1f}%  "
                    f"dark_05={dark_05 / dtype_max * 100:.1f}%"
                )

                if bright_99 > bright_thresh:
                    # Clipping — reduce exposure.
                    hi = exposure
                elif dark_05 < dark_thresh and exposure >= MAX_EXPOSURE_US:
                    # Too dark even at max exposure for this gain — try next gain.
                    break
                elif dark_05 < dark_thresh:
                    # Too dark — increase exposure.
                    lo = exposure
                else:
                    # Both constraints satisfied.
                    self.get_logger().info(
                        f"{cam}: converged — "
                        f"exposure={exposure} µs  gain={gain:.1f}"
                    )
                    return exposure, gain

                exposure = (lo + hi) // 2
                self._set_params(cam, [_param("ExposureTime", exposure)])

            # If the inner loop ended without a break and the last frame was
            # actually acceptable (e.g. hit MAX_ITERS in a good region), return.
            if dark_05 >= dark_thresh and bright_99 <= bright_thresh:
                self.get_logger().info(
                    f"{cam}: converged (max iters) — "
                    f"exposure={exposure} µs  gain={gain:.1f}"
                )
                return exposure, gain

            final_exposure = exposure
            final_gain = gain

        # Exhausted all gain levels — lock at last settings.
        self.get_logger().warn(
            f"{cam}: scene too dark at all gain levels — "
            f"locking at exposure={final_exposure} µs  gain={final_gain:.1f}. "
            "Check lighting conditions."
        )
        return final_exposure, final_gain

    # -----------------------------------------------------------------------
    # Main calibration sequence (background thread)
    # -----------------------------------------------------------------------

    def _cal_task(self) -> None:
        if not self._force_cal:
            self.get_logger().info(
                f"auto_cal: waiting for drone to clear {ALT_THRESHOLD_M} m AGL "
                "before starting exposure calibration and irradiance reference capture."
            )
        while not self._above_alt.wait(timeout=5.0):
            self.get_logger().warn(
                f"auto_cal: still waiting for {ALT_THRESHOLD_M} m AGL — "
                "exposure and irradiance calibration have NOT run yet. "
                "Cameras are in auto-exposure mode."
            )
        self.get_logger().info(
            f"Drone above {ALT_THRESHOLD_M} m — starting auto-calibration"
        )

        # Calibrate cam0 and cam1 sequentially (independent cameras, different lenses).
        for cam in ("cam0", "cam1"):
            exp_us, gain = self._calibrate(cam)
            # Final lock: confirm AeEnable=False with the found settings.
            self._set_params(cam, [
                _param("AeEnable", False),
                _param("ExposureTime", exp_us),
                _param("AnalogueGain", gain),
            ])
            self.get_logger().info(
                f"{cam} locked — ExposureTime={exp_us} µs  AnalogueGain={gain:.1f}×"
            )

        # Publish irradiance reference for stream_processor.
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

        # Signal panel_scan that exposure is locked and it can start imaging.
        self._pub_exposure_locked.publish(Bool(data=True))
        self.get_logger().info(
            "Exposure locked — panel_scan may now begin QR detection."
        )
        self.get_logger().info("Auto-calibration complete.")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    # MultiThreadedExecutor lets the background calibration thread make
    # service calls while the main thread keeps spinning callbacks.
    executor = rclpy.executors.MultiThreadedExecutor()
    node = AutoCalNode()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
