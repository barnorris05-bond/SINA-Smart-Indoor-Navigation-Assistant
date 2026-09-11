"""
depth/validity.py

Robust depth measurement and validity assessment.

Pure NumPy logic — no DepthAI, no camera, no I/O. Everything here is
unit-testable without hardware.

Design (documented per Phase 2 requirements):

- Depth images are uint16 millimeters. Invalid pixels are exactly 0
  (StereoDepth outputs 0 where disparity/depth could not be computed:
  out of range, occlusions, left-right-check failures, confidence
  rejections, textureless regions).
- We additionally clamp to a configured physical range [DEPTH_MIN_MM,
  DEPTH_MAX_MM]; values outside are treated as invalid.
- Region measurement uses robust statistics over the *central* portion of
  a bounding box, avoiding edge pixels where stereo mixing between
  foreground/background produces garbage:
      median over the central ROI of valid pixels.
  Median rejects both outlier shadows (too-near reflections) and holes.
- A region is rejected outright when the valid-pixel fraction or the
  valid-pixel count is too low: unknown is NOT treated as safe.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from config.camera import (
    DEPTH_MIN_MM,
    DEPTH_MAX_MM,
    DEPTH_MIN_VALID_RATIO,
    DEPTH_MIN_SAMPLES,
)

# Central fraction of each bbox axis used for measurement (edge-stripping).
ROI_CENTER_FRACTION = 0.5


@dataclass
class DepthStats:
    """Outcome of a robust depth measurement attempt."""

    distance_mm: Optional[int]      # Robust distance, None if rejected.
    valid_pixels: int               # Valid px inside the measured ROI.
    total_pixels: int               # Total px inside the measured ROI.
    valid_ratio: float              # valid_pixels / total_pixels (0 if empty).
    reason: Optional[str] = None    # Rejection reason for logging/tests.

    @property
    def distance_m(self) -> Optional[float]:
        return None if self.distance_mm is None else self.distance_mm / 1000.0

    @property
    def valid(self) -> bool:
        return self.distance_mm is not None


def invalid_mask(depth_mm: np.ndarray) -> np.ndarray:
    """
    Boolean mask of INVALID pixels for a uint16 depth frame (mm):
    zero pixels plus values outside the configured physical range.
    """
    if depth_mm.ndim != 2:
        raise ValueError(f"depth frame must be 2-D (H, W); got {depth_mm.shape}")
    values = depth_mm.astype(np.int32, copy=False)
    return (values <= 0) | (values < DEPTH_MIN_MM) | (values > DEPTH_MAX_MM)


def central_roi(
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
    fraction: float = ROI_CENTER_FRACTION,
) -> Tuple[int, int, int, int]:
    """
    Shrink a bbox to its central part and clamp it to the frame.

    bbox is (x1, y1, x2, y2); frame_shape is (H, W). Returns a possibly
    empty-but-well-ordered (x1, y1, x2, y2) with x1<x2 and y1<y2 strictly
    guaranteed by expanding half a pixel if the shrink collapses.
    """
    x1, y1, x2, y2 = bbox
    frame_h, frame_w = frame_shape

    # Defensive ordering (callers may pass unordered boxes).
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    w = x2 - x1
    h = y2 - y1
    cut_x = int(w * (1.0 - fraction) / 2.0)
    cut_y = int(h * (1.0 - fraction) / 2.0)

    cx1 = x1 + cut_x
    cy1 = y1 + cut_y
    cx2 = x2 - cut_x
    cy2 = y2 - cut_y

    # Guarantee strictly positive extent after shrink.
    if cx2 - cx1 < 1:
        cx1, cx2 = x1, max(x1 + 1, x2)
    if cy2 - cy1 < 1:
        cy1, cy2 = y1, max(y1 + 1, y2)

    # Clamp to frame.
    cx1 = max(0, min(cx1, frame_w - 1))
    cx2 = max(cx1 + 1, min(cx2, frame_w))
    cy1 = max(0, min(cy1, frame_h - 1))
    cy2 = max(cy1 + 1, min(cy2, frame_h))
    return cx1, cy1, cx2, cy2


def measure_region(
    depth_mm: np.ndarray,
    bbox: Tuple[int, int, int, int],
    min_valid_ratio: float = DEPTH_MIN_VALID_RATIO,
    min_samples: int = DEPTH_MIN_SAMPLES,
) -> DepthStats:
    """
    Robustly measure the distance of a bbox region in a depth frame.

    Method (documented choice for Phase 2/3):
      1. Take the CENTRAL ROI of the bbox (edge-stripping).
      2. Discard invalid pixels (0 / out-of-range) inside it.
      3. Require min_samples valid px AND min_valid_ratio coverage.
      4. Return the MEDIAN of valid pixels (robust to outliers/holes).
    """
    x1, y1, x2, y2 = bbox
    frame_h, frame_w = depth_mm.shape[:2]

    # Fully off-frame boxes cannot be measured (clamping below would
    # otherwise silently collapse them onto a corner pixel).
    if x2 <= 0 or y2 <= 0 or x1 >= frame_w or y1 >= frame_h:
        return DepthStats(None, 0, 0, 0.0, reason="empty_roi")

    rx1, ry1, rx2, ry2 = central_roi(bbox, depth_mm.shape)
    roi = depth_mm[ry1:ry2, rx1:rx2]
    total = int(roi.size)

    if total == 0:
        return DepthStats(None, 0, 0, 0.0, reason="empty_roi")

    bad = invalid_mask(roi)
    valid_values = roi[~bad]
    valid = int(valid_values.size)
    ratio = valid / total

    if valid < min_samples:
        return DepthStats(
            None, valid, total, ratio,
            reason="too_few_samples" if valid > 0 else "no_valid_pixels",
        )
    if ratio < min_valid_ratio:
        return DepthStats(None, valid, total, ratio, reason="low_valid_ratio")

    distance_mm = int(np.median(valid_values))
    return DepthStats(distance_mm, valid, total, ratio)


def validate_depth_frame(depth_mm: np.ndarray) -> dict:
    """
    Frame-level sanity report used by hardware tests and health checks.
    Returns validity ratio and basic stats WITHOUT any bbox logic.
    """
    if depth_mm is None or depth_mm.size == 0:
        return {
            "ok": False, "valid_ratio": 0.0,
            "valid_pixels": 0, "total_pixels": 0,
            "reason": "no_frame",
        }

    bad = invalid_mask(depth_mm)
    total = int(depth_mm.size)
    valid = int((~bad).sum())
    ratio = valid / total
    return {
        "ok": ratio > 0.0,
        "valid_ratio": ratio,
        "valid_pixels": valid,
        "total_pixels": total,
        "reason": None if ratio > 0.0 else "all_invalid",
    }
