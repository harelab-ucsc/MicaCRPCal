"""
Tests for mica_crp_cal.auto_cal_node pure functions.

All tests are self-contained — no ROS2 init required.
"""

import numpy as np
import pytest
from rcl_interfaces.msg import ParameterType

from mica_crp_cal.auto_cal_node import (
    BRIGHT_CEIL,
    DARK_FLOOR,
    GAIN_STEPS,
    MAX_EXPOSURE_US,
    MIN_EXPOSURE_US,
    NUM_CAM0_SLICES,
    _analyze_cam0,
    _analyze_cam1,
    _param,
)


# ---------------------------------------------------------------------------
# _analyze_cam0
# ---------------------------------------------------------------------------

class TestAnalyzeCam0:

    def _make_frame(self, slice_values, slice_w=320, h=200, dtype=np.uint16):
        """Build a synthetic cam0 frame with uniform DN per slice."""
        w = slice_w * NUM_CAM0_SLICES
        img = np.zeros((h, w), dtype=dtype)
        for i, val in enumerate(slice_values):
            img[:, i * slice_w: (i + 1) * slice_w] = val
        return img

    def test_uniform_frame_returns_same_bright_and_dark(self):
        """Uniform DN → p99 == p05 == pixel value."""
        img = self._make_frame([32000] * 4)
        bright, dark = _analyze_cam0(img)
        assert bright == pytest.approx(32000, abs=1)
        assert dark == pytest.approx(32000, abs=1)

    def test_bright_is_max_across_slices(self):
        """bright_99 is the 99th pct of the BRIGHTEST slice."""
        img = self._make_frame([10000, 20000, 30000, 60000])
        bright, _ = _analyze_cam0(img)
        assert bright == pytest.approx(60000, abs=100)

    def test_dark_is_min_across_slices(self):
        """dark_05 is the 5th pct of the DARKEST slice."""
        img = self._make_frame([1000, 20000, 30000, 40000])
        _, dark = _analyze_cam0(img)
        assert dark == pytest.approx(1000, abs=100)

    def test_all_zero_frame(self):
        img = self._make_frame([0, 0, 0, 0])
        bright, dark = _analyze_cam0(img)
        assert bright == 0.0
        assert dark == 0.0

    def test_all_saturated_16bit(self):
        img = self._make_frame([65535, 65535, 65535, 65535])
        bright, dark = _analyze_cam0(img)
        assert bright == pytest.approx(65535, abs=1)
        assert dark == pytest.approx(65535, abs=1)

    def test_returns_floats(self):
        img = self._make_frame([10000] * 4)
        bright, dark = _analyze_cam0(img)
        assert isinstance(bright, float)
        assert isinstance(dark, float)

    def test_bright_always_gte_dark(self):
        """bright_99 ≥ dark_05 by definition."""
        for vals in [[1000, 2000, 3000, 4000], [5000] * 4, [0, 0, 0, 65535]]:
            img = self._make_frame(vals)
            bright, dark = _analyze_cam0(img)
            assert bright >= dark

    def test_8bit_frame_works(self):
        img = self._make_frame([50, 100, 150, 200], dtype=np.uint8)
        bright, dark = _analyze_cam0(img)
        assert bright == pytest.approx(200, abs=2)
        assert dark == pytest.approx(50, abs=2)


# ---------------------------------------------------------------------------
# _analyze_cam1
# ---------------------------------------------------------------------------

class TestAnalyzeCam1:
    """
    cam1 uses SBGGR16 Bayer: B at (even row, even col), G at (even,odd)+(odd,even),
    R at (odd row, odd col).  _analyze_cam1 returns worst-case percentiles across
    the three channels so a clipped R or B isn't hidden by dominant green pixels.
    """

    def _make_bayer(self, b_val, g_val, r_val, h=200, w=320):
        """Build a synthetic SBGGR Bayer frame with uniform per-channel values."""
        img = np.zeros((h, w), dtype=np.uint16)
        img[0::2, 0::2] = b_val   # B
        img[0::2, 1::2] = g_val   # Gr
        img[1::2, 0::2] = g_val   # Gb
        img[1::2, 1::2] = r_val   # R
        return img

    def test_uniform_frame(self):
        img = self._make_bayer(40000, 40000, 40000)
        bright, dark = _analyze_cam1(img)
        assert bright == pytest.approx(40000, abs=1)
        assert dark == pytest.approx(40000, abs=1)

    def test_bright_is_max_across_channels(self):
        """bright_99 is from the brightest channel, not the whole-frame pct."""
        img = self._make_bayer(b_val=10000, g_val=30000, r_val=60000)
        bright, _ = _analyze_cam1(img)
        assert bright == pytest.approx(60000, abs=1)

    def test_dark_is_min_across_channels(self):
        """dark_05 is from the darkest channel, not the whole-frame pct."""
        img = self._make_bayer(b_val=2000, g_val=30000, r_val=30000)
        _, dark = _analyze_cam1(img)
        assert dark == pytest.approx(2000, abs=1)

    def test_clipped_red_not_masked_by_green(self):
        """
        A clipped R channel (65535) must show up in bright_99 even though G
        has moderate values.  This is the case a whole-frame percentile would
        miss when R pixels are only 25% of the frame.
        """
        img = self._make_bayer(b_val=20000, g_val=30000, r_val=65535)
        bright, _ = _analyze_cam1(img)
        assert bright == pytest.approx(65535, abs=1)

    def test_dark_blue_not_masked_by_green(self):
        """A very dark B channel must show up in dark_05."""
        img = self._make_bayer(b_val=100, g_val=30000, r_val=30000)
        _, dark = _analyze_cam1(img)
        assert dark == pytest.approx(100, abs=2)

    def test_all_zero(self):
        img = np.zeros((100, 100), dtype=np.uint16)
        bright, dark = _analyze_cam1(img)
        assert bright == 0.0
        assert dark == 0.0

    def test_returns_tuple_of_floats(self):
        img = self._make_bayer(1000, 1000, 1000)
        result = _analyze_cam1(img)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_bright_gte_dark(self):
        img = self._make_bayer(5000, 25000, 50000)
        bright, dark = _analyze_cam1(img)
        assert bright >= dark


# ---------------------------------------------------------------------------
# _param helper
# ---------------------------------------------------------------------------

class TestParam:

    def test_bool_type(self):
        p = _param("AeEnable", True)
        assert p.name == "AeEnable"
        assert p.value.type == ParameterType.PARAMETER_BOOL
        assert p.value.bool_value is True

    def test_bool_false(self):
        p = _param("AeEnable", False)
        assert p.value.bool_value is False

    def test_int_type(self):
        p = _param("ExposureTime", 5000)
        assert p.name == "ExposureTime"
        assert p.value.type == ParameterType.PARAMETER_INTEGER
        assert p.value.integer_value == 5000

    def test_float_type(self):
        p = _param("AnalogueGain", 2.5)
        assert p.name == "AnalogueGain"
        assert p.value.type == ParameterType.PARAMETER_DOUBLE
        assert p.value.double_value == pytest.approx(2.5)

    def test_int_coerced_from_float_passes_as_double(self):
        """Non-bool, non-int Python value → PARAMETER_DOUBLE."""
        p = _param("AnalogueGain", 4.0)
        assert p.value.type == ParameterType.PARAMETER_DOUBLE


# ---------------------------------------------------------------------------
# Exposure binary search invariants
# ---------------------------------------------------------------------------

class TestBinarySearchInvariants:
    """
    Verify that the algorithm constants and termination conditions are coherent.
    These tests don't run the full binary search (that requires camera hardware)
    but check that the bounds and targets make physical sense.
    """

    def test_max_exposure_within_frame_duration(self):
        """MAX_EXPOSURE_US must be well below the 200 ms frame duration (199977 µs)."""
        assert MAX_EXPOSURE_US < 199977

    def test_min_exposure_positive(self):
        assert MIN_EXPOSURE_US > 0

    def test_min_less_than_max(self):
        assert MIN_EXPOSURE_US < MAX_EXPOSURE_US

    def test_gain_steps_monotonically_increasing(self):
        for a, b in zip(GAIN_STEPS, GAIN_STEPS[1:]):
            assert b > a

    def test_gain_steps_start_at_one(self):
        assert GAIN_STEPS[0] == pytest.approx(1.0)

    def test_bright_ceil_above_dark_floor(self):
        assert BRIGHT_CEIL > DARK_FLOOR

    def test_bright_ceil_below_1(self):
        """Must leave headroom to avoid clipping."""
        assert BRIGHT_CEIL < 1.0

    def test_dark_floor_above_0(self):
        """Must require some minimum signal."""
        assert DARK_FLOOR > 0.0

    def test_termination_condition_coherent(self):
        """
        Simulate one binary search step: if bright_99 is below BRIGHT_CEIL
        and dark_05 is above DARK_FLOOR, the algorithm should accept.
        """
        dtype_max = 65535.0
        bright_99 = BRIGHT_CEIL * dtype_max * 0.9   # 10% below ceiling
        dark_05 = DARK_FLOOR * dtype_max * 1.5      # 50% above floor
        in_range = (bright_99 <= BRIGHT_CEIL * dtype_max
                    and dark_05 >= DARK_FLOOR * dtype_max)
        assert in_range

    def test_clipping_condition(self):
        dtype_max = 65535.0
        bright_99 = BRIGHT_CEIL * dtype_max * 1.01  # just over ceiling
        clipping = bright_99 > BRIGHT_CEIL * dtype_max
        assert clipping

    def test_too_dark_condition(self):
        dtype_max = 65535.0
        dark_05 = DARK_FLOOR * dtype_max * 0.5  # below floor
        too_dark = dark_05 < DARK_FLOOR * dtype_max
        assert too_dark
