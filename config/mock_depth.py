"""
config/mock_depth.py

MOCK DEPTH CONFIGURATION (Phase B) — DEVELOPMENT/TEST VALUES.

!!! NOT CALIBRATED — NOT REAL MEASUREMENTS !!!

Every distance produced while this module is active is SIMULATED.
It exists so the fusion / navigation / audio layers can be developed
and regression-tested while real stereo depth is blocked (USB 2.0,
stereo hardware validation pending). All values here must be replaced
or re-calibrated once OAK-D stereo depth is validated (Phase E/F).

Nothing in this file affects the real stereo path in camera/ or the
robust-measurement logic in depth/validity.py.
"""

# ==========================================================
# Depth provider selection
# ==========================================================
# True  -> mock providers (ScenarioDepthProvider). Current mode.
# False -> OakStereoDepthProvider (real OAK-D stereo, Phase E).
USE_MOCK_DEPTH = True

# Which mock strategy to use: "scenario" (recommended) or "bbox".
MOCK_MODE = "scenario"

# ==========================================================
# Strategy A: Scenario table (RECOMMENDED for current stage)
# ==========================================================
# Deterministic, human-specified distances keyed by
# (label_lower, region), falling back to per-label defaults.
# Distance in METERS. Development/test values — NOT calibrated.

SCENARIO_DISTANCES_M = {
    # Center lane — drives STOP/SLOW_DOWN distance decisions
    ("person", "CENTER"): 1.2,
    ("chair", "CENTER"): 2.5,
    ("door", "CENTER"): 3.5,
    ("table", "CENTER"): 2.2,
    ("stairs", "CENTER"): 2.0,
    # Lateral lanes — drive MOVE_LEFT / MOVE_RIGHT decisions
    ("person", "LEFT"): 2.0,
    ("person", "RIGHT"): 1.8,
    ("chair", "LEFT"): 1.5,
    ("chair", "RIGHT"): 1.5,
    ("table", "LEFT"): 2.0,
    ("table", "RIGHT"): 2.0,
}

# Fallback when (label, region) is not in the table above.
# Covers the common COCO classes SINA prioritizes; anything else stays
# UNAVAILABLE (the mock never invents numbers for unknown objects).
LABEL_DEFAULT_DISTANCES_M = {
    "person": 2.0,
    "stairs": 2.0,
    "chair": 2.0,
    "door": 3.0,
    "table": 2.0,
    "bottle": 2.5,
}

# ==========================================================
# Strategy B: Bounding-box synthetic distance (pinhole model)
# ==========================================================
# distance_m = (assumed_real_height_m * focal_px) / bbox_height_px
#
# These are SYNTHETIC ASSUMPTIONS, not measurements:
# - FOCAL_PX approximates a ~65 deg horizontal FOV at 640 px width
#   (f = (640/2) / tan(65/2 deg) ~= 484 px). Real value differs
#   per device calibration — fine for testing, NOT for real use.
# - Real heights are coarse averages (adult ~1.7 m, chair ~0.9 m).
BBOX_FOCAL_PX = 484.0

BBOX_REAL_HEIGHT_M = {
    "person": 1.7,
    "chair": 0.9,
    "table": 0.75,
    "door": 2.0,
    "bottle": 0.25,
    "laptop": 0.25,
    "book": 0.25,
}

# Focal-length fallback for labels without an assumed height.
BBOX_DEFAULT_HEIGHT_M = 0.5

# Clamp range for synthetic distances (matches the physical depth
# validity range in config/camera.py).
BBOX_MIN_M = 0.2
BBOX_MAX_M = 20.0
