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

import multiprocessing as mp
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray

WINDOW_SIZE = 30    # sliding window length in frames (~10 s at 3 Hz)
CONFIRM_HITS = 3    # detections required within the window
CONFIRM_FRAMES = CONFIRM_HITS   # alias used by tests and docstring
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

    Tries all four cardinal directions (below, above, right, left of the QR)
    and returns the projection that is most likely to contain the panel.
    Direction selection is purely geometric (no image data) so this is used
    as a fallback when edge detection fails.

    Args:
        pts: (4, 2) float32 QR corner points in any order.

    Returns:
        (4, 2) float32 panel corners.
    """
    pts = pts.reshape(4, 2).astype(np.float32)

    def _build(axis: int, positive: bool) -> np.ndarray:
        order = np.argsort(pts[:, axis])
        if positive:
            near_two, far_two = pts[order[:2]], pts[order[2:]]
        else:
            near_two, far_two = pts[order[2:]], pts[order[:2]]
        step = far_two.mean(axis=0) - near_two.mean(axis=0)
        perp = 1 - axis
        far_l, far_r = far_two[np.argsort(far_two[:, perp])]
        gap = step * PANEL_GAP_FRAC
        tl = far_l + gap
        tr = far_r + gap
        return np.array(
            [tl, tr, tr + step * PANEL_SIZE_FRAC, tl + step * PANEL_SIZE_FRAC],
            dtype=np.float32,
        )

    # Default to projecting below (original assumption); used as fallback.
    return _build(1, True)


def _find_panel_by_edges(
    band: np.ndarray,
    hint_center: tuple | None = None,
    hint_radius: float | None = None,
) -> np.ndarray | None:
    """Locate the reflective panel ROI via edge detection and contour analysis.

    Applies Canny edge detection, finds external contours, and scores each
    roughly-square candidate by mean pixel brightness. The white reflective
    panel is the brightest roughly-square object in the scene.

    Args:
        band:         (H, W) image slice (any bit depth).
        hint_center:  (cx, cy) pixel coordinate to constrain the search.
                      Candidates whose centre falls outside hint_radius are
                      skipped. Pass None to search the full image.
        hint_radius:  Search radius in pixels around hint_center.

    Returns:
        (4, 2) float32 oriented bounding-box corners of the best candidate,
        or None if no plausible panel rectangle is found.
    """
    u8 = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) \
        if band.dtype != np.uint8 else band.copy()

    blurred = cv2.GaussianBlur(u8, (7, 7), 0)
    edges = cv2.Canny(blurred, 20, 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = band.shape[:2]
    # Panel spans roughly 3–40 % of the slice width.
    min_area = (w * 0.03) ** 2
    max_area = (w * 0.40) ** 2

    best_roi, best_score = None, -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw < 1 or rh < 1:
            continue
        # Reject elongated shapes — the panel is roughly square.
        if max(rw, rh) / min(rw, rh) > 2.0:
            continue

        if hint_center is not None and hint_radius is not None:
            cx, cy = rect[0]
            dx, dy = cx - hint_center[0], cy - hint_center[1]
            if dx * dx + dy * dy > hint_radius ** 2:
                continue

        box = cv2.boxPoints(rect).astype(np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(box).astype(np.int32), 255)
        pixels = band[mask > 0]
        if pixels.size == 0:
            continue
        score = float(np.mean(pixels.astype(np.float64)))
        if score > best_score:
            best_score = score
            best_roi = box

    return best_roi


_QR_UPSCALE = 2          # upscale factor passed to QReader to improve small-QR detection
_QR_MIN_CONFIDENCE = 0.3  # lower than default 0.5 — we're far away, accept weaker hits


def _qreader_worker(in_q: mp.Queue, out_q: mp.Queue) -> None:
    """Runs in a separate process — loads QReader once, processes frames."""
    from qreader import QReader
    detector = QReader(min_confidence=_QR_MIN_CONFIDENCE)
    while True:
        item = in_q.get()
        if item is None:
            break
        frame_id, slices = item  # slices: list of (slice_idx, gray uint8)
        found_corners = None
        found_slices = []
        for s, gray in slices:
            try:
                # Upscale before detection so the QR occupies more pixels in
                # the YOLO input — critical when hovering at 3 m AGL where the
                # QR code is only ~20-40 px wide in a single band slice.
                h, w = gray.shape[:2]
                up = cv2.resize(
                    gray,
                    (w * _QR_UPSCALE, h * _QR_UPSCALE),
                    interpolation=cv2.INTER_LINEAR,
                )
                texts, dets = detector.detect_and_decode(
                    image=up, return_detections=True
                )
                if texts and dets:
                    # Scale detected bbox back to original resolution.
                    x1, y1, x2, y2 = (v / _QR_UPSCALE for v in dets[0]['bbox_xyxy'])
                    pts = np.array(
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        dtype=np.float32,
                    )
                    if found_corners is None:
                        found_corners = pts
                    found_slices.append(s)
            except Exception:
                pass
        out_q.put((frame_id, found_corners, found_slices))


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
        self._window: deque = deque(maxlen=WINDOW_SIZE)
        self._frame_id: int = 0
        self._first_frame_saved: bool = False
        self._last_qr_pts = None          # corners from last confirmed detection
        self._last_detected_slices: list = []

        # QReader process — created lazily in _exposure_locked_cb so YOLO
        # doesn't consume CPU during auto_cal's binary search.
        self._qr_ctx = mp.get_context('spawn')
        self._qr_in: mp.Queue | None = None
        self._qr_out: mp.Queue | None = None
        self._qr_proc = None
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

        # Start the QReader worker now — no point loading YOLO during auto_cal.
        self._qr_in = self._qr_ctx.Queue(maxsize=2)
        self._qr_out = self._qr_ctx.Queue()
        self._qr_proc = self._qr_ctx.Process(
            target=_qreader_worker,
            args=(self._qr_in, self._qr_out),
            daemon=True,
        )
        self._qr_proc.start()

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

        # Save the very first frame after exposure lock so the user can verify quality.
        if not self._first_frame_saved:
            self._first_frame_saved = True
            try:
                import os
                vis = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                out = os.path.join(
                    os.path.expanduser("~/parsed_flight"),
                    "panel_scan_first_frame.png")
                cv2.imwrite(out, vis)
                self.get_logger().info(f"First scan frame saved: {out}")
            except Exception as e:
                self.get_logger().warn(f"Could not save first frame: {e}")

        h, w = raw.shape[:2]
        slice_w = w // NUM_SLICES

        if self._qr_in is None:
            return  # worker not started yet

        # Submit slices 1+2 to the QReader worker process (non-blocking).
        slices = []
        for s in (1, 2):
            band = raw[:, s * slice_w: (s + 1) * slice_w]
            gray = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) \
                if raw.dtype != np.uint8 else band.copy()
            slices.append((s, gray))
        try:
            self._qr_in.put_nowait((self._frame_id, slices))
        except Exception:
            pass  # queue full — skip this frame
        self._frame_id += 1

        # Drain any completed results from the worker.
        while not self._qr_out.empty():
            try:
                _, qr_corners, detected_slices = self._qr_out.get_nowait()
            except Exception:
                break

            bboxes: list[np.ndarray | None] = [
                qr_corners.reshape(1, 4, 2) if qr_corners is not None else None
                for _ in range(NUM_SLICES)
            ]
            hit = qr_corners is not None
            self._window.append(hit)
            hits = sum(self._window)

            if hit:
                self._last_raw = raw
                self._last_bboxes = bboxes
                self._last_qr_pts = qr_corners
                self._last_detected_slices = detected_slices
                self.get_logger().info(
                    f"QR located in slice(s) {detected_slices} "
                    f"({hits}/{CONFIRM_HITS} in last {len(self._window)} frames)"
                )
            if hits >= CONFIRM_HITS:
                self._publish_calibration()
                return

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

        qr_pts = self._last_qr_pts

        # Use the 735 nm slice (best contrast) to find the panel via edge
        # detection. Search within 3× the QR bounding-box width of the QR
        # centre so we don't pick up a bright object elsewhere in the scene.
        ref_band = raw[:, 2 * slice_w: 3 * slice_w]
        qr_cx = float(qr_pts[:, 0].mean())
        qr_cy = float(qr_pts[:, 1].mean())
        qr_size = float(max(
            qr_pts[:, 0].max() - qr_pts[:, 0].min(),
            qr_pts[:, 1].max() - qr_pts[:, 1].min(),
        ))
        projected_panel = _find_panel_by_edges(
            ref_band,
            hint_center=(qr_cx, qr_cy),
            hint_radius=qr_size * 3.0,
        )
        if projected_panel is None:
            self.get_logger().warn(
                "Edge detection found no panel — falling back to QR projection."
            )
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

        self._save_debug_image(raw, factors, slice_w, projected_panel)
        rclpy.shutdown()

    def _save_debug_image(
        self,
        raw: np.ndarray,
        factors: list,
        slice_w: int,
        panel_proj: np.ndarray,
    ) -> None:
        """Save an 8-bit BGR image with QR and panel ROI boxes drawn per slice."""
        try:
            import os

            # Normalise 16-bit → 8-bit and convert to BGR for drawing.
            vis = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

            qr_pts = self._last_qr_pts
            detected = self._last_detected_slices
            for i in range(NUM_SLICES):
                x_off = i * slice_w
                label = f"{CAM0_BAND_NM[i]}nm  f={factors[i]:.3f}"
                cv2.putText(vis_bgr, label, (x_off + 10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

                panel_global = panel_proj.copy()
                panel_global[:, 0] += x_off
                cv2.polylines(vis_bgr, [np.round(panel_global).astype(np.int32)],
                              isClosed=True, color=(255, 80, 0), thickness=4)

                # QR box only where it was actually detected.
                if i in detected:
                    qr_global = qr_pts.copy()
                    qr_global[:, 0] += x_off
                    cv2.polylines(vis_bgr, [np.round(qr_global).astype(np.int32)],
                                  isClosed=True, color=(0, 255, 0), thickness=4)

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
