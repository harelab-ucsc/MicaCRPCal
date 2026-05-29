"""
MicaCRPCal panel scan node.

Subscribes to the cam0 image stream (the only nadir-pointing camera).
cam0 is a 4-band multispectral image: four band images laid side-by-side
horizontally, each occupying one quarter of the full frame width.

Trigger
-------
The node waits for /cal/exposure_locked (published by auto_cal once cameras
are locked at flight exposure settings) before opening the QR scan window.
This guarantees the panel is imaged at the same ExposureTime and AnalogueGain
used for all flight images — critical because mean_panel_DN encodes the
exposure time and a mismatch would scale all reflectance values incorrectly.

The 30-second scan window starts when /cal/exposure_locked is received.
Place the CRP panel flat on the ground directly below the hovering drone.

Algorithm
---------
1. Wait for /cal/exposure_locked from auto_cal.
2. Run QR detection on all four cam0 band slices simultaneously for up to
   SCAN_TIMEOUT_S seconds. Each slice uses its own corners where detected;
   slices that cannot decode the QR (e.g. NIR at 850 nm has poor ink
   contrast) fall back to corners from a detecting slice.
3. When the QR tag is confirmed across CONFIRM_FRAMES consecutive frames,
   derive the flat reflective panel ROI from the QR corner geometry.
   The panel is directly below the QR code in the holder and is the same size.
4. For each of the four cam0 band slices, compute the mean raw DN over the
   panel ROI at full sensor bit-depth.
5. Look up the panel's certified spectral albedo at each band wavelength
   (interpolated from the MicaSense-supplied CRP CSV).
6. Publish four per-band calibration factors on /panel_cal/irradiance (latched):
       factor[i] = albedo(λ_i) / (mean_panel_DN[i] / dtype_max)

stream_processor passes these four factors directly to the C++ spectral_correct
extension, which applies per pixel:
    corrected = (raw_DN / dtype_max) * factor  →  true reflectance ∈ [0, 1]
"""

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
# No extra QR library needed — cv2.QRCodeDetectorAruco is built into OpenCV 4.5+
# and handles perspective distortion much better than QRCodeDetector.
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray

WINDOW_SIZE = 30   # sliding window length in frames (~10 s at 3 Hz)
CONFIRM_HITS = 3   # detections required within the window
NUM_SLICES = 4

# If no QR detected after this many seconds, start cycling through exposures.
FALLBACK_TIMEOUT_S = 30.0
FALLBACK_STEP_S = 5.0       # seconds between each exposure change
# Exposures to try (µs) — low first to fight saturation, then up.
FALLBACK_EXPOSURES = [500, 1000, 2000, 4000, 8000, 16000, 33000]

# Center wavelengths (nm) for cam0 band slices 0-3.
CAM0_BAND_NM = (450, 695, 735, 850)

# Panel geometry relative to the QR bounding quad.
# Physical holder dimensions (RP06 CRP target):
#   QR code:  83 mm × 83 mm
#   Gap:      36 mm  (QR bottom edge to panel top)
#   Panel:   100 mm × 100 mm
# PANEL_GAP_FRAC  = 36 / 83  ≈ 0.434
# PANEL_SIZE_FRAC = 100 / 83 ≈ 1.205
PANEL_GAP_FRAC = 0.434
PANEL_SIZE_FRAC = 1.205

# Default CRP CSV — bundled alongside this file.
_DEFAULT_CSV = Path(__file__).parent / "data" / "RP06-2120405-OB.csv"

_LATCHED_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    depth=1,
)


def _load_albedo(csv_path: str | Path, wavelengths_nm: tuple) -> list:
    """Interpolate panel albedo at the requested wavelengths from a MicaSense CRP CSV.

    The CSV has two columns: wavelength_nm, reflectance (no header).
    """
    data = np.loadtxt(str(csv_path), delimiter=",")
    wl = data[:, 0]
    refl = data[:, 1]
    return [float(np.interp(nm, wl, refl)) for nm in wavelengths_nm]


def _panel_roi_from_qr(pts: np.ndarray) -> np.ndarray:
    """Return the flat panel ROI corners given the four QR corner points.

    The panel holder has the QR code at one end and the flat reflective panel
    at the other end (same size, directly adjacent). We project one QR-height
    past the QR's image-space bottom edge to locate the panel.

    This works for any in-plane rotation where the panel is below the QR in
    image-Y (Y=0 at top). If the holder is flipped 180° so the panel is above
    the QR in the image, re-orient the holder and rescan.

    Args:
        pts: (4, 2) float32 QR corner points in any order.

    Returns:
        (4, 2) float32 panel corners in [TL, TR, BR, BL] image order.
    """
    pts = pts.reshape(4, 2).astype(np.float32)

    # Split into "top" (smaller image-Y) and "bottom" (larger image-Y) pairs.
    order = np.argsort(pts[:, 1])
    top_two = pts[order[:2]]
    bot_two = pts[order[2:]]

    step = bot_two.mean(axis=0) - top_two.mean(axis=0)  # QR height vector

    # Sort each pair left-to-right for consistent corner labelling.
    bot_l, bot_r = bot_two[np.argsort(bot_two[:, 0])]

    gap = step * PANEL_GAP_FRAC
    panel_tl = bot_l + gap
    panel_tr = bot_r + gap
    panel_bl = panel_tl + step * PANEL_SIZE_FRAC
    panel_br = panel_tr + step * PANEL_SIZE_FRAC

    return np.array([panel_tl, panel_tr, panel_br, panel_bl], dtype=np.float32)


def _snap_to_corners(
    band: np.ndarray,
    projected: np.ndarray,
    search_radius: int = 40,
    max_corners: int = 20,
    quality: float = 0.05,
    min_dist: float = 10.0,
) -> np.ndarray:
    """Snap each projected panel corner to the nearest strong image corner.

    For each of the 4 projected panel ROI corners, searches a small
    neighbourhood in the band image for strong corners using
    goodFeaturesToTrack. If a detected corner falls within `search_radius`
    pixels it replaces the projected point; otherwise the projected point
    is kept as fallback.

    Args:
        band:          (H, W) uint8 grayscale slice image.
        projected:     (4, 2) float32 projected panel corners.
        search_radius: pixel radius around each projected corner to search.

    Returns:
        (4, 2) float32 snapped panel corners.
    """
    h, w = band.shape[:2]
    snapped = projected.copy()

    for i, proj in enumerate(projected):
        px, py = int(proj[0]), int(proj[1])

        # Clamp search window to image bounds.
        x0 = max(0, px - search_radius)
        y0 = max(0, py - search_radius)
        x1 = min(w, px + search_radius)
        y1 = min(h, py + search_radius)

        patch = band[y0:y1, x0:x1]
        if patch.size == 0:
            continue

        corners = cv2.goodFeaturesToTrack(
            patch, maxCorners=max_corners, qualityLevel=quality,
            minDistance=min_dist,
        )
        if corners is None:
            continue

        # Convert patch-local coords back to full-image coords.
        corners_global = corners.reshape(-1, 2) + np.array([x0, y0], dtype=np.float32)

        # Find the nearest detected corner to the projected point.
        dists = np.linalg.norm(corners_global - proj, axis=1)
        nearest_idx = int(np.argmin(dists))
        if dists[nearest_idx] < search_radius:
            snapped[i] = corners_global[nearest_idx]

    return snapped


class PanelScanNode(Node):

    def __init__(self):
        super().__init__("panel_scan")
        self._bridge = CvBridge()
        self._detector = cv2.QRCodeDetectorAruco()
        self._window: deque = deque(maxlen=WINDOW_SIZE)  # sliding detection window
        self._done = False
        self._last_raw: np.ndarray | None = None
        self._last_bboxes: list[np.ndarray | None] = [None] * NUM_SLICES

        # Fallback: if no QR detected after FALLBACK_TIMEOUT_S, start cycling
        # through FALLBACK_EXPOSURES on cam0 every FALLBACK_STEP_S seconds.
        self._scan_open_t: float | None = None   # set when exposure locked
        self._fallback_idx: int = 0
        self._fallback_last_t: float = 0.0
        self._cam0_param_client = self.create_client(
            SetParameters, "/cam0/camera_node/set_parameters"
        )

        # Exposure-lock gate: QR scanning only starts once auto_cal has locked
        # the cameras. mean_panel_DN encodes ExposureTime — scanning before lock
        # would produce factors calibrated at a different exposure than flight.
        self.declare_parameter("force_cal", False)
        self._force_cal: bool = self.get_parameter("force_cal").value
        # Always wait for /cal/exposure_locked — even in force_cal mode.
        # force_cal skips the altitude gate on auto_cal side, but the panel must
        # still be scanned at the locked exposure or the factors will be wrong.
        self._exposure_locked: bool = False

        # CRP albedo CSV — can be overridden via ROS parameter.
        self.declare_parameter("crp_csv", str(_DEFAULT_CSV))
        csv_path = self.get_parameter("crp_csv").value

        try:
            self._albedo = _load_albedo(csv_path, CAM0_BAND_NM)
        except Exception as ex:
            self.get_logger().fatal(f"Failed to load CRP albedo CSV ({csv_path}): {ex}")
            raise

        self.get_logger().info(
            f"CRP albedo loaded from {csv_path}: "
            + ", ".join(
                f"{nm}nm={a:.4f}" for nm, a in zip(CAM0_BAND_NM, self._albedo)
            )
        )

        self._pub = self.create_publisher(
            Float32MultiArray, "/panel_cal/irradiance", _LATCHED_QOS
        )

        # cam0 is the only nadir-pointing camera.
        self.create_subscription(
            Image,
            "/cam0/camera_node/image_raw",
            self._img_cb,
            10,
        )

        # Wait for auto_cal to lock exposure before scanning — ensures panel DN
        # is measured at the same ExposureTime used for flight images.
        self.create_subscription(
            Bool,
            "/cal/exposure_locked",
            self._exposure_locked_cb,
            _LATCHED_QOS,
        )

        self.create_timer(2.0, self._watchdog)

        if self._force_cal:
            self.get_logger().warn(
                "force_cal=True — still waiting for /cal/exposure_locked from "
                "auto_cal before scanning. Panel scan always runs at the locked "
                "exposure so factors are valid for flight images."
            )
            self.get_logger().warn(
                "PRE-FLIGHT CHECK: The calibration panel MUST be in direct "
                "sunlight with NO shadow on the reflective surface."
            )
        else:
            self.get_logger().info(
                "panel_scan: waiting for auto_cal to complete exposure "
                "calibration (/cal/exposure_locked) before opening scan window."
            )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _exposure_locked_cb(self, msg: Bool) -> None:
        if self._exposure_locked:
            return
        self._exposure_locked = True
        self._scan_open_t = time.monotonic()
        self.get_logger().info(
            "Exposure locked — panel scan window open. "
            "Place the CRP panel flat on the ground directly below the drone "
            "with the QR tag visible. Panel must be in direct sunlight with "
            "NO shadow on the reflective surface."
        )

    def _img_cb(self, msg: Image) -> None:
        if self._done or not self._exposure_locked:
            return

        try:
            raw = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as ex:
            self.get_logger().warn(
                f"imgmsg_to_cv2 failed: {ex}", throttle_duration_sec=5.0
            )
            return

        # Save the very first frame so the user can verify image quality.
        if not self._window:
            try:
                import os
                vis = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                out = os.path.join(os.path.expanduser("~/parsed_flight"), "panel_scan_first_frame.png")
                cv2.imwrite(out, vis)
                self.get_logger().info(f"First scan frame saved: {out}")
            except Exception as e:
                self.get_logger().warn(f"Could not save first frame: {e}")

        h, w = raw.shape[:2]
        slice_w = w // NUM_SLICES

        # Run pyzbar on slices 1+2 (695nm, 735nm) — best ink contrast.
        # Scanning all 4 slices per frame at 3Hz is too slow; 0=450nm and
        # 3=850nm rarely detect and block the callback. The panel ROI is
        # projected from the found corners and applied to all 4 slices at
        # calibration time regardless of which slice detected here.
        qr_corners: np.ndarray | None = None
        detected_slices: list[int] = []

        for s in (1, 2):
            band = raw[:, s * slice_w: (s + 1) * slice_w]
            gray = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) \
                if raw.dtype != np.uint8 else band.copy()
            try:
                data, bbox, _ = self._detector.detectAndDecode(gray)
            except Exception as e:
                self.get_logger().error(f"QR detect failed on slice {s}: {e}")
                data, bbox = "", None
            if data and bbox is not None:
                pts = bbox.reshape(4, 2).astype(np.float32)
                if qr_corners is None:
                    qr_corners = pts
                detected_slices.append(s)
                self.get_logger().debug(f"Slice {s}: QR decoded — '{data}'")

        # Build bboxes for all slices using the shared corners.
        bboxes: list[np.ndarray | None] = [
            qr_corners.reshape(1, 4, 2) if qr_corners is not None else None
            for _ in range(NUM_SLICES)
        ]
        detected_data = qr_corners is not None

        hit = bool(detected_data)  # detected_data is now a bool, not None/str
        self._window.append(hit)
        hits = sum(self._window)

        if hit:
            self._last_raw = raw
            self._last_bboxes = bboxes
            self.get_logger().info(
                f"QR located in slice(s) {detected_slices} "
                f"({hits}/{CONFIRM_HITS} in last {len(self._window)} frames)"
            )
        if hits >= CONFIRM_HITS:
            self._publish_calibration()

    def _watchdog(self) -> None:
        if self._done:
            return
        if not self._exposure_locked:
            self.get_logger().warn(
                "panel_scan: still waiting for auto_cal to publish "
                "/cal/exposure_locked — QR scan has not started yet."
            )
            return
        self.get_logger().warn(
            "panel_scan: QR tag not yet detected — still scanning. "
            "Ensure the CRP panel is flat below the drone, QR tag visible, "
            "in direct sunlight with no shadow."
        )

        # After FALLBACK_TIMEOUT_S with no detections, cycle through exposures.
        if self._scan_open_t is None:
            return
        elapsed = time.monotonic() - self._scan_open_t
        if elapsed < FALLBACK_TIMEOUT_S or sum(self._window) > 0:
            return
        now = time.monotonic()
        if now - self._fallback_last_t < FALLBACK_STEP_S:
            return

        exp = FALLBACK_EXPOSURES[self._fallback_idx % len(FALLBACK_EXPOSURES)]
        self._fallback_idx += 1
        self.get_logger().warn(
            f"panel_scan: no QR detected after {elapsed:.0f} s — "
            f"trying cam0 ExposureTime={exp} µs "
            f"(step {self._fallback_idx}/{len(FALLBACK_EXPOSURES)})"
        )
        self._fallback_last_t = now

        if not self._cam0_param_client.service_is_ready():
            return
        req = SetParameters.Request()
        req.parameters = [Parameter(
            name="ExposureTime",
            value=ParameterValue(
                type=ParameterType.PARAMETER_INTEGER,
                integer_value=exp,
            ),
        )]
        self._cam0_param_client.call_async(req)

    # ------------------------------------------------------------------
    # Calibration factor computation
    # ------------------------------------------------------------------

    def _publish_calibration(self) -> None:
        self._done = True

        raw = self._last_raw
        bboxes = self._last_bboxes
        if raw is None or all(b is None for b in bboxes):
            self.get_logger().error(
                "No valid frame stored — cannot compute calibration."
            )
            rclpy.shutdown()
            return

        h, w = raw.shape[:2]
        slice_w = w // NUM_SLICES

        # dtype_max for normalisation: 65535 for 16-bit, 255 for 8-bit.
        if np.issubdtype(raw.dtype, np.integer):
            dtype_max = float(np.iinfo(raw.dtype).max)
        else:
            dtype_max = 1.0

        # Only calibrate slices that can see QR ink (DETECT_SLICES = 1, 2).
        # Dark/NIR bands (0=450nm, 3=850nm) get factor=1.0 — no projection.
        qr_pts = next(
            b.reshape(4, 2).astype(np.float32) for b in bboxes if b is not None
        )

        # Project once — shared across all slices.
        projected_panel = _panel_roi_from_qr(qr_pts)

        factors = []
        for i in range(NUM_SLICES):
            band = raw[:, i * slice_w: (i + 1) * slice_w]

            band_u8 = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) \
                if band.dtype != np.uint8 else band
            panel_pts = _snap_to_corners(band_u8, projected_panel)
            panel_pts_int = np.round(panel_pts).astype(np.int32)

            mask = np.zeros(band.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, panel_pts_int, 255)

            pixels = band[mask > 0]
            if pixels.size == 0:
                self.get_logger().error(
                    f"Slice {i} ({CAM0_BAND_NM[i]} nm): panel mask is empty. "
                    "Check panel position. "
                    "Falling back to factor=1.0."
                )
                factors.append(1.0)
                continue

            mean_dn = float(np.mean(pixels.astype(np.float64)))
            if mean_dn <= 0.0:
                self.get_logger().error(
                    f"Slice {i} ({CAM0_BAND_NM[i]} nm): mean panel DN is zero — "
                    "possible dead camera. Falling back to factor=1.0."
                )
                factors.append(1.0)
                continue

            # factor = albedo / (panel_DN / dtype_max)
            # C++ applies: corrected = (pixel_DN / dtype_max) * factor
            #            = (pixel_DN / panel_DN) * albedo  →  true reflectance
            # Irradiance at scan time is encoded in panel_DN; no separate
            # spectrometer correction is applied (panel may be in shade).
            factor = self._albedo[i] / (mean_dn / dtype_max)
            factors.append(factor)
            self.get_logger().info(
                f"Slice {i} ({CAM0_BAND_NM[i]} nm): "
                f"mean_DN={mean_dn:.1f}  albedo={self._albedo[i]:.4f}  "
                f"factor={factor:.4f}"
            )

        msg = Float32MultiArray()
        msg.data = [float(f) for f in factors]
        self._pub.publish(msg)
        self.get_logger().info(
            f"Panel calibration published ({NUM_SLICES} bands) on /panel_cal/irradiance"
        )

        self._save_debug_image(raw, bboxes, factors, slice_w, fallback_idx)
        rclpy.shutdown()

    def _save_debug_image(
        self,
        raw: np.ndarray,
        bboxes: list,
        factors: list,
        slice_w: int,
        fallback_idx: int,
    ) -> None:
        """Save an 8-bit BGR image with QR and panel ROI boxes drawn per slice."""
        try:
            import os

            # Normalise 16-bit → 8-bit and convert to BGR for drawing.
            vis = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

            panel_proj = _panel_roi_from_qr(qr_pts)
            for i in range(NUM_SLICES):
                x_off = i * slice_w
                label = f"{CAM0_BAND_NM[i]}nm  f={factors[i]:.3f}"
                cv2.putText(vis_bgr, label, (x_off + 10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

                qr_global = qr_pts.copy();  qr_global[:, 0] += x_off
                panel_global = panel_proj.copy();  panel_global[:, 0] += x_off

                cv2.polylines(vis_bgr, [np.round(qr_global).astype(np.int32)],
                              isClosed=True, color=(0, 255, 0), thickness=4)
                cv2.polylines(vis_bgr, [np.round(panel_global).astype(np.int32)],
                              isClosed=True, color=(255, 80, 0), thickness=4)

            out_dir = os.path.expanduser("~/parsed_flight")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "panel_cal_debug.png")
            cv2.imwrite(out_path, vis_bgr)
            self.get_logger().info(f"Debug image saved: {out_path}")
        except Exception as e:
            self.get_logger().warn(f"Failed to save debug image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PanelScanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
