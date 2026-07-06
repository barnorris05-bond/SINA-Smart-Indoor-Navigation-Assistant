"""
Camera configuration settings for OAK-D Lite.
"""

import depthai as dai

# RGB Camera Settings
RGB_WIDTH = 1280
RGB_HEIGHT = 720
RGB_FPS = 30

# Mono Cameras (for stereo)
MONO_WIDTH = 1280
MONO_HEIGHT = 720
DEPTH_FPS = 30

# Stereo Settings
STEREO_PRESET = dai.node.StereoDepth.PresetMode.FAST_DENSITY
ENABLE_LEFT_RIGHT_CHECK = True

# Queue Settings
QUEUE_SIZE = 4
BLOCKING_QUEUE = False
