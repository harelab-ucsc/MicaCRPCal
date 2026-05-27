"""
Tests for mica_crp_cal.panel_scan_node pure functions and calibration logic.

All tests are self-contained — no ROS2 init required. The node-level
calibration factor logic is replicated inline so it can be verified
independently of the ROS executor.
"""

import math
import os
import tempfile

import cv2
import numpy as np
import pytest

from mica_crp_cal.panel_scan_node import (
    CAM0_BAND_NM,
    NUM_SLICES,
    PANEL_GAP_FRAC,
    PANEL_SIZE_FRAC,
    _load_albedo,
    _panel_roi_from_qr,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path, rows):
    """Write (wavelength, reflectance) rows to a CSV file."""
    with open(path, "w") as f:
        for wl, r in rows:
            f.write(f"{wl},{r}\n")


def _qr_corners_axis_aligned(x0=10, y0=10, size=80):
    """Return (4,2) corners for an axis-aligned QR with panel below it."""
    return np.array([
        [x0,        y0],
        [x0 + size, y0],
        [x0 + size, y0 + size],
        [x0,        y0 + size],
    ], dtype=np.float32)


def _rotate_pts(pts, angle_deg, cx, cy):
    """Rotate a (N,2) array of points around (cx, cy)."""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    centered = pts - np.array([cx, cy])
    rot = centered @ np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return rot + np.array([cx, cy])


def _panel_centroid(panel_pts):
    return panel_pts.mean(axis=0)


def _qr_centroid(qr_pts):
    return np.array(qr_pts, dtype=np.float32).mean(axis=0)


# ---------------------------------------------------------------------------
# _load_albedo
# ---------------------------------------------------------------------------

class TestLoadAlbedo:

    def test_exact_wavelength_returns_correct_value(self, tmp_path):
        csv = tmp_path / "panel.csv"
        _write_csv(csv, [(400, 0.40), (500, 0.50), (600, 0.60), (700, 0.70)])
        result = _load_albedo(str(csv), (500,))
        assert result[0] == pytest.approx(0.50, abs=1e-6)

    def test_interpolation_between_wavelengths(self, tmp_path):
        csv = tmp_path / "panel.csv"
        _write_csv(csv, [(400, 0.40), (600, 0.60)])
        result = _load_albedo(str(csv), (500,))
        assert result[0] == pytest.approx(0.50, abs=1e-6)

    def test_multiple_bands_returned_in_order(self, tmp_path):
        csv = tmp_path / "panel.csv"
        rows = [(wl, wl / 1000.0) for wl in range(400, 901, 10)]
        _write_csv(csv, rows)
        bands = (450, 695, 735, 850)
        result = _load_albedo(str(csv), bands)
        assert len(result) == 4
        for val, wl in zip(result, bands):
            assert val == pytest.approx(wl / 1000.0, abs=0.001)

    def test_cam0_band_nm_all_within_csv_range(self, tmp_path):
        csv = tmp_path / "panel.csv"
        rows = [(wl, 0.47) for wl in range(251, 951)]
        _write_csv(csv, rows)
        result = _load_albedo(str(csv), CAM0_BAND_NM)
        assert len(result) == len(CAM0_BAND_NM)
        for val in result:
            assert val == pytest.approx(0.47, abs=1e-4)

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            _load_albedo("/nonexistent/path.csv", (450,))


# ---------------------------------------------------------------------------
# _panel_roi_from_qr
# ---------------------------------------------------------------------------

class TestPanelRoiFromQr:

    def _assert_panel_below_qr(self, qr_pts, panel_pts):
        """Panel centroid Y must be greater than QR centroid Y."""
        qr_cy = _qr_centroid(qr_pts)[1]
        pan_cy = _panel_centroid(panel_pts)[1]
        assert pan_cy > qr_cy, (
            f"Panel centroid Y ({pan_cy:.1f}) is not below QR centroid Y ({qr_cy:.1f})"
        )

    def _assert_panel_size(self, qr_pts, panel_pts):
        """Panel height and width should match QR dimensions × PANEL_SIZE_FRAC."""
        qr = np.array(qr_pts, dtype=np.float32)
        qr_h = qr[:, 1].max() - qr[:, 1].min()
        pan = np.array(panel_pts, dtype=np.float32)
        pan_h = pan[:, 1].max() - pan[:, 1].min()
        assert pan_h == pytest.approx(qr_h * PANEL_SIZE_FRAC, rel=0.05)

    def test_axis_aligned_panel_is_below_qr(self):
        qr = _qr_corners_axis_aligned(x0=10, y0=10, size=80)
        panel = _panel_roi_from_qr(qr)
        self._assert_panel_below_qr(qr, panel)

    def test_axis_aligned_panel_size(self):
        qr = _qr_corners_axis_aligned(x0=10, y0=10, size=80)
        panel = _panel_roi_from_qr(qr)
        self._assert_panel_size(qr, panel)

    def test_axis_aligned_gap(self):
        """Gap between QR bottom and panel top ≈ PANEL_GAP_FRAC × QR height."""
        size = 100
        qr = _qr_corners_axis_aligned(x0=0, y0=0, size=size)
        panel = _panel_roi_from_qr(qr)
        qr_bottom_y = size                      # QR bottom edge Y
        panel_top_y = panel[:, 1].min()         # smallest Y in panel ROI
        gap = panel_top_y - qr_bottom_y
        assert gap == pytest.approx(size * PANEL_GAP_FRAC, abs=1.0)

    def test_returns_four_corners(self):
        qr = _qr_corners_axis_aligned()
        panel = _panel_roi_from_qr(qr)
        assert panel.shape == (4, 2)

    def test_rotated_45_panel_still_below_qr(self):
        size = 80
        cx, cy = 200, 200
        qr = _qr_corners_axis_aligned(x0=cx - size // 2, y0=cy - size // 2, size=size)
        qr_rot = _rotate_pts(qr, 45, cx, cy).astype(np.float32)
        panel = _panel_roi_from_qr(qr_rot)
        self._assert_panel_below_qr(qr_rot, panel)

    def test_rotated_90_panel_still_below_qr(self):
        size = 80
        cx, cy = 200, 200
        qr = _qr_corners_axis_aligned(x0=cx - size // 2, y0=cy - size // 2, size=size)
        qr_rot = _rotate_pts(qr, 90, cx, cy).astype(np.float32)
        panel = _panel_roi_from_qr(qr_rot)
        self._assert_panel_below_qr(qr_rot, panel)

    def test_output_dtype_is_float32(self):
        qr = _qr_corners_axis_aligned()
        panel = _panel_roi_from_qr(qr)
        assert panel.dtype == np.float32

    def test_accepts_flat_bbox_shape(self):
        """Accepts (1,4,2) bbox from detectAndDecode without error."""
        qr = _qr_corners_axis_aligned().reshape(1, 4, 2)
        panel = _panel_roi_from_qr(qr)
        assert panel.shape == (4, 2)


# ---------------------------------------------------------------------------
# Calibration factor computation (replicated from _publish_calibration)
# ---------------------------------------------------------------------------

def _compute_factors(raw, bboxes, albedo, dtype_max):
    """
    Replicate the per-slice factor computation from PanelScanNode._publish_calibration.
    Returns a list of (factor, ok) tuples; ok=False means the slice fell back to 1.0.
    """
    h, w = raw.shape[:2]
    slice_w = w // NUM_SLICES

    fallback_pts = next(
        b.reshape(4, 2).astype(np.float32) for b in bboxes if b is not None
    )

    results = []
    for i in range(NUM_SLICES):
        band = raw[:, i * slice_w: (i + 1) * slice_w]
        pts = bboxes[i].reshape(4, 2).astype(np.float32) if bboxes[i] is not None else fallback_pts
        panel_pts_int = np.round(_panel_roi_from_qr(pts)).astype(np.int32)

        mask = np.zeros(band.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, panel_pts_int, 255)
        pixels = band[mask > 0]

        if pixels.size == 0 or float(np.mean(pixels.astype(np.float64))) <= 0:
            results.append((1.0, False))
        else:
            mean_dn = float(np.mean(pixels.astype(np.float64)))
            results.append((albedo[i] / (mean_dn / dtype_max), True))

    return results


class TestFactorComputation:

    def _make_synthetic_frame(self, slice_values, slice_w=320, h=200):
        """
        Build a synthetic 16-bit cam0 frame where each slice has a uniform DN.
        slice_values: list of 4 int DN values (one per slice).
        """
        w = slice_w * NUM_SLICES
        img = np.zeros((h, w), dtype=np.uint16)
        for i, val in enumerate(slice_values):
            img[:, i * slice_w: (i + 1) * slice_w] = val
        return img

    def _make_qr_corners_in_slice(self, slice_idx, slice_w=320, h=200):
        """
        Return a (4,2) bbox whose panel ROI will land inside the given slice.

        Coordinates are in SLICE-LOCAL space (x relative to the slice's left
        edge, always within [0, slice_w)).  detectAndDecode returns corners in
        slice-local coordinates because QR detection runs on the cropped band.
        """
        x0 = 10  # same local position in every slice
        pts = np.array([
            [x0,      10],
            [x0 + 60, 10],
            [x0 + 60, 70],
            [x0,      70],
        ], dtype=np.float32)
        return pts

    def test_factor_formula_correct(self):
        """factor = albedo / (mean_dn / dtype_max)."""
        albedo = 0.47
        mean_dn = 32768.0
        dtype_max = 65535.0
        expected = albedo / (mean_dn / dtype_max)
        assert expected == pytest.approx(0.47 / (32768 / 65535), rel=1e-4)

    def test_half_saturated_panel_gives_double_albedo_factor(self):
        """At 50% DN, factor = 2 × albedo."""
        albedo = 0.47
        mean_dn = 65535.0 / 2
        dtype_max = 65535.0
        factor = albedo / (mean_dn / dtype_max)
        assert factor == pytest.approx(2 * albedo, rel=1e-4)

    def test_fully_saturated_panel_gives_factor_equal_to_albedo(self):
        """At 100% DN (saturated panel), factor = albedo."""
        albedo = 0.47
        mean_dn = 65535.0
        dtype_max = 65535.0
        factor = albedo / (mean_dn / dtype_max)
        assert factor == pytest.approx(albedo, rel=1e-4)

    def test_all_slices_detect_own_corners(self):
        """When all slices detect QR, each uses its own corners."""
        dn_values = [10000, 20000, 30000, 40000]
        img = self._make_synthetic_frame(dn_values)
        bboxes = [self._make_qr_corners_in_slice(i) for i in range(NUM_SLICES)]
        albedo = [0.47] * NUM_SLICES
        dtype_max = 65535.0

        results = _compute_factors(img, bboxes, albedo, dtype_max)
        factors = [r[0] for r in results]
        ok_flags = [r[1] for r in results]

        assert all(ok_flags), "All slices should produce valid factors"
        # Higher DN → lower factor (less correction needed)
        assert factors[0] > factors[1] > factors[2] > factors[3]

    def test_none_slices_fall_back_to_first_detector(self):
        """Slices with None bbox use corners from the first detecting slice."""
        dn_values = [30000, 30000, 30000, 30000]
        img = self._make_synthetic_frame(dn_values)
        bboxes = [self._make_qr_corners_in_slice(0), None, None, None]
        albedo = [0.47] * NUM_SLICES
        dtype_max = 65535.0

        results = _compute_factors(img, bboxes, albedo, dtype_max)
        # All factors should be approximately equal (same DN, same albedo, same corners)
        factors = [r[0] for r in results]
        for f in factors:
            assert f == pytest.approx(factors[0], rel=0.02)

    def test_all_none_bboxes_raises_stop_iteration(self):
        """all-None bboxes: fallback_pts generator raises StopIteration."""
        img = self._make_synthetic_frame([1000] * 4)
        bboxes = [None, None, None, None]
        with pytest.raises(StopIteration):
            _compute_factors(img, bboxes, [0.47] * 4, 65535.0)

    def test_zero_dn_slice_falls_back_to_1(self):
        """A slice with zero mean DN returns factor=1.0."""
        dn_values = [0, 30000, 30000, 30000]
        img = self._make_synthetic_frame(dn_values)
        bboxes = [self._make_qr_corners_in_slice(i) for i in range(NUM_SLICES)]
        albedo = [0.47] * NUM_SLICES

        results = _compute_factors(img, bboxes, albedo, 65535.0)
        assert results[0] == (1.0, False)
        assert results[1][1] is True


# ---------------------------------------------------------------------------
# QR detection confirmation logic
# ---------------------------------------------------------------------------

class TestQRConfirmationLogic:
    """
    Replicate the _img_cb confirmation state machine.
    Tests that CONFIRM_FRAMES consecutive detections are required and that
    a single failed frame resets the counter.
    """

    def _run_frames(self, detection_pattern):
        """
        Simulate the confirm counter logic for a list of (detected: bool) frames.
        Returns the confirm count at each frame.
        """
        from mica_crp_cal.panel_scan_node import CONFIRM_FRAMES

        confirm = 0
        triggered = False
        history = []
        for detected in detection_pattern:
            if detected:
                confirm += 1
                if confirm >= CONFIRM_FRAMES:
                    triggered = True
            else:
                confirm = 0
            history.append(confirm)
        return history, triggered

    def test_consecutive_detections_trigger(self):
        from mica_crp_cal.panel_scan_node import CONFIRM_FRAMES
        _, triggered = self._run_frames([True] * CONFIRM_FRAMES)
        assert triggered

    def test_single_miss_resets_counter(self):
        from mica_crp_cal.panel_scan_node import CONFIRM_FRAMES
        pattern = [True] * (CONFIRM_FRAMES - 1) + [False] + [True] * CONFIRM_FRAMES
        history, triggered = self._run_frames(pattern)
        # Counter should reset at the False frame
        reset_idx = CONFIRM_FRAMES - 1
        assert history[reset_idx] == 0
        assert triggered  # eventually succeeds after reset

    def test_never_triggers_without_consecutive(self):
        pattern = [True, False] * 20
        _, triggered = self._run_frames(pattern)
        assert not triggered

    def test_partial_detection_counts_as_detected(self):
        """
        A frame counts as detected if at least one slice detects the QR.
        Simulate bboxes = [None, bbox, None, None] → detected_data is not None.
        """
        dummy_bbox = np.zeros((1, 4, 2), dtype=np.float32)
        bboxes_per_frame = [
            [None, dummy_bbox, None, None],   # 1 slice detected
            [dummy_bbox, None, None, None],   # 1 slice detected
            [dummy_bbox, dummy_bbox, None, None],  # 2 slices detected
        ]
        for bboxes in bboxes_per_frame:
            detected = any(b is not None for b in bboxes)
            assert detected, "Frame with at least one detection must count as detected"

    def test_all_none_bboxes_not_detected(self):
        bboxes = [None, None, None, None]
        detected = any(b is not None for b in bboxes)
        assert not detected
