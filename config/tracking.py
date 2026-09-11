"""
config/tracking.py

OBJECT TRACKING CONFIGURATION (Phase 5).

!!! DEVELOPMENT/TEST TUNING — NOT CALIBRATED !!!

These values were chosen for the current development setup (640x400 RGB
@ 15 FPS, YOLO11n detections, indoor ranges). They are NOT derived from
real-world tracking evaluation and are NOT safety-calibrated. Recalibrate
against recorded real sequences before drawing any conclusions from them.
"""

# Frames a track may stay unmatched (MISSING) before it is EXPIRED and
# removed from the active registry. At ~15 FPS this is ~0.7 s: enough to
# ride out brief YOLO flicker (e.g. a person dropping below the 0.50
# confidence gate for a few frames) without spawning a new ID, short
# enough that a truly gone object is released quickly.
MAX_MISSED_FRAMES = 10

# Bounding-box IoU at/above which two boxes are considered the same
# object (matching pass 1). Kept low on purpose: single-frame detection
# jitter on small/near objects can halve apparent overlap.
MIN_IOU = 0.15

# Maximum centroid displacement in pixels (640x400 frame) for matching
# pass 2 — covers meaningful motion where box overlap is small or zero
# (e.g. a person walking: centers shift far more than boxes overlap).
MAX_CENTROID_DISTANCE = 80.0
