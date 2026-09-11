"""
tests/test_depth_validity.py

No-hardware unit tests for depth/validity.py — pure NumPy logic.

Verifies: invalid masking (zero/out-of-range), central ROI extraction,
median-based region measurement, rejection thresholds (min samples,
valid ratio), and frame-level validation. Deterministic; runs in ms.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth.validity import (
    DepthStats,
    central_roi,
    invalid_mask,
    measure_region,
    validate_depth_frame,
)
from config.camera import DEPTH_MIN_MM, DEPTH_MAX_MM


class TestInvalidMask(unittest.TestCase):
    def test_zero_pixels_are_invalid(self):
        depth = np.array([[0, 1500], [3000, 0]], dtype=np.uint16)
        bad = invalid_mask(depth)
        self.assertEqual(bad.tolist(), [[True, False], [False, True]])

    def test_in_range_pixels_are_valid(self):
        depth = np.array([[DEPTH_MIN_MM, 1500], [DEPTH_MAX_MM, 1000]], dtype=np.uint16)
        bad = invalid_mask(depth)
        self.assertEqual(bad.tolist(), [[False, False], [False, False]])

    def test_out_of_range_values_are_invalid(self):
        depth = np.array(
            [[DEPTH_MIN_MM - 1, DEPTH_MAX_MM + 1], [0, 1500]], dtype=np.uint16
        )
        bad = invalid_mask(depth)
        self.assertTrue(bad[0, 0])
        self.assertTrue(bad[0, 1])
        self.assertFalse(bad[1, 1])

    def test_rejects_non_2d_input(self):
        with self.assertRaises(ValueError):
            invalid_mask(np.zeros((3, 4, 3), dtype=np.uint16))


class TestCentralROI(unittest.TestCase):
    def test_center_half_of_box(self):
        # 100x100 box at (0,0): central 50% -> (25,25,75,75)
        roi = central_roi((0, 0, 100, 100), (400, 640))
        self.assertEqual(roi, (25, 25, 75, 75))

    def test_clamps_to_frame(self):
        roi = central_roi((600, 350, 700, 450), (400, 640))
        x1, y1, x2, y2 = roi
        self.assertTrue(0 <= x1 < x2 <= 640)
        self.assertTrue(0 <= y1 < y2 <= 400)

    def test_handles_degenerate_boxes(self):
        # Degenerate input must not produce an empty ROI.
        roi = central_roi((100, 100, 100, 100), (400, 640))
        x1, y1, x2, y2 = roi
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)

    def test_unordered_coords_repaired(self):
        roi = central_roi((200, 300, 100, 150), (400, 640))
        x1, y1, x2, y2 = roi
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)


class TestMeasureRegion(unittest.TestCase):
    def test_clean_region_returns_median(self):
        depth = np.full((400, 640), 2500, dtype=np.uint16)  # 2.5 m everywhere
        # Small outlier INSIDE the ROI (bbox (280,160,360,240) -> central
        # ROI rows 180:220, cols 300:340): median must ignore it.
        depth[195:205, 315:325] = 40  # 40mm reflection artifact
        stats = measure_region(depth, (280, 160, 360, 240))
        self.assertTrue(stats.valid)
        self.assertEqual(stats.distance_mm, 2500)
        self.assertEqual(stats.reason, None)

    def test_all_zero_region_rejected(self):
        depth = np.zeros((400, 640), dtype=np.uint16)
        stats = measure_region(depth, (280, 160, 360, 240))
        self.assertFalse(stats.valid)
        self.assertEqual(stats.reason, "no_valid_pixels")

    def test_too_few_samples_rejected(self):
        depth = np.zeros((400, 640), dtype=np.uint16)
        # Sprinkle a few valid px: below DEPTH_MIN_SAMPLES.
        depth[200, 320] = 1500
        depth[200, 321] = 1500
        depth[201, 320] = 1500
        stats = measure_region(depth, (280, 160, 360, 240))
        self.assertFalse(stats.valid)
        self.assertEqual(stats.reason, "too_few_samples")

    def test_low_valid_ratio_rejected(self):
        depth = np.zeros((400, 640), dtype=  np.uint16)
        # Fill only a tiny corner of the ROI with valid values: ratio < 0.2.
        depth[190:200, 310:320] = 1500
        stats = measure_region(depth, (280, 160, 360, 240))
        self.assertFalse(stats.valid)
        self.assertEqual(stats.reason, "low_valid_ratio")

    def test_mixed_noise_median_is_robust(self):
        rng = np.random.default_rng(42)
        depth = rng.integers(2450, 2560, size=(400, 640)).astype(np.uint16)
        stats = measure_region(depth, (280, 160, 360, 240))
        self.assertTrue(stats.valid)
        self.assertTrue(2450 <= stats.distance_mm <= 2560)
        self.assertEqual(stats.valid_ratio, 1.0)

    def test_distance_m_property(self):
        depth = np.full((400, 640), 1400, dtype=np.uint16)
        stats = measure_region(depth, (280, 160, 360, 240))
        self.assertAlmostEqual(stats.distance_m, 1.4, places=3)

    def test_empty_roi_rejected(self):
        depth = np.full((400, 640), 1400, dtype=np.uint16)
        stats = measure_region(depth, (-50, -50, -10, -10))
        self.assertFalse(stats.valid)
        self.assertEqual(stats.reason, "empty_roi")


class TestValidateDepthFrame(unittest.TestCase):
    def test_full_valid_frame(self):
        depth = np.full((400, 640), 3000, dtype=np.uint16)
        report = validate_depth_frame(depth)
        self.assertTrue(report["ok"])
        self.assertEqual(report["valid_ratio"], 1.0)
        self.assertEqual(report["valid_pixels"], 400 * 640)

    def test_zero_frame_is_not_ok(self):
        report = validate_depth_frame(np.zeros((400, 640), dtype=np.uint16))
        self.assertFalse(report["ok"])
        self.assertEqual(report["reason"], "all_invalid")
        self.assertEqual(report["valid_ratio"], 0.0)

    def test_none_frame_handled(self):
        report = validate_depth_frame(None)
        self.assertFalse(report["ok"])
        self.assertEqual(report["reason"], "no_frame")

    def test_partial_frame(self):
        depth = np.zeros((400, 640), dtype=np.uint16)
        depth[:200, :] = 3000  # half valid
        report = validate_depth_frame(depth)
        self.assertTrue(report["ok"])
        self.assertAlmostEqual(report["valid_ratio"], 0.5, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
