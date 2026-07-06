"""
tests/inspect_stereo_presets.py
Inspect available StereoDepth presets in DepthAI v3.
"""

import depthai as dai

print("=" * 60)
print("StereoDepth Preset Modes Inspection")
print("=" * 60)

preset_enum = dai.node.StereoDepth.PresetMode

print("Enum class:", preset_enum)
print()

# Try to list members
try:
    presets = list(preset_enum)
    print("Available Presets:")
    for p in presets:
        print(f"  - {p}")
except Exception as e:
    print("Could not list directly:", e)
    print("\nUsing dir() instead:")
    print(dir(preset_enum))