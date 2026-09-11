"""
Camera configuration settings for OAK-D Lite.

NOTE: intentionally DepthAI-import-free at module level. The depth
package (depth/validity.py) and the whole mock-depth stack import this
module; requiring DepthAI here would make the mock path dependent on
hardware libraries. Stereo enum values are provided lazily via
get_stereo_preset() / get_stereo_align_socket() and are only needed by
the gated stereo branch (ENABLE_STEREO_HARDWARE=True).
"""

# RGB Camera Settings
RGB_WIDTH = 640
RGB_HEIGHT = 400
RGB_FPS = 15

# ==========================================================
# Stereo HARDWARE gate (Phase E switch)
# ==========================================================
# False (current): CameraManager builds the RGB-only pipeline and never
#   attempts stereo init. Mock-depth development mode — the previously
#   crashing stereo pipeline is NOT started, per hardware-blocked status.
# True:  the aligned stereo-depth branch is built (Phase E hardware
#   validation via tests/test_depth_stream.py, requires the USB 3.x link).
# Pairing rule: Phase E = ENABLE_STEREO_HARDWARE=True AND
#   config/mock_depth.USE_MOCK_DEPTH=False.
ENABLE_STEREO_HARDWARE = False

# Mono Cameras (stereo pair feeding StereoDepth)
#
# Phase 2 decision: run mono at the SAME size/rate as RGB (640x400@15).
# Rationale:
#   - depth frames aligned to CAM_A then come out pixel-aligned with the
#     640x400 RGB frame (required for Phase 3 object-depth fusion),
#   - a unified 15 FPS simplifies RGB/depth synchronization,
#   - roughly 4x lower USB bandwidth than 1280x720@30 (the device
#     previously showed X_LINK instability at the high profile).
# Tradeoff: coarser disparity resolution than 720p; acceptable for
# near-field indoor ranging. 1280x720@30 remains the known-good
# high-detail fallback if more depth accuracy is ever required.
MONO_WIDTH = 640
MONO_HEIGHT = 480
DEPTH_FPS = 15

# Stereo Settings
# (enum values resolved lazily — see module NOTE above)
ENABLE_LEFT_RIGHT_CHECK = True


def get_stereo_preset():
    """StereoDepth preset enum (imports DepthAI lazily, stereo path only)."""
    import depthai as dai
    return dai.node.StereoDepth.PresetMode.FAST_DENSITY


def get_stereo_align_socket():
    """Depth alignment socket (imports DepthAI lazily, stereo path only)."""
    import depthai as dai
    return dai.CameraBoardSocket.CAM_A

# Stereo post-processing (v3: set on stereo.initialConfig, NOT the node).
# Confidence threshold 0-255: higher rejects more low-confidence disparity
# pixels (they become invalid 0), trading density for reliability.
STEREO_CONFIDENCE_THRESHOLD = 200
# Depth validity range in millimeters for indoor assistive navigation.
# OAK-D Lite practical near limit ~150-200mm; cap far noise at 20m.
DEPTH_MIN_MM = 200
DEPTH_MAX_MM = 20000

# If fewer than this fraction of a region's pixels are valid, the depth
# measurement for that region is rejected outright (depth/validity.py).
DEPTH_MIN_VALID_RATIO = 0.2
# Minimum valid pixel count inside a region before trusting statistics.
DEPTH_MIN_SAMPLES = 30

# Depth stream warmup: device AE/stereo needs a moment after start before
# measurements are meaningful (frames).
DEPTH_WARMUP_FRAMES = 20

# Queue Settings
QUEUE_SIZE = 4
BLOCKING_QUEUE = False
