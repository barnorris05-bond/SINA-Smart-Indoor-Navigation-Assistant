"""
tests/inspect_camera_outputs.py
Inspect available outputs from the unified Camera node in DepthAI v3.
"""

import depthai as dai

print("=" * 60)
print("Camera Node Outputs Inspection")
print("=" * 60)

pipeline = dai.Pipeline()

# Using CAM_B as per your request (you can change to CAM_A if needed)
cam = pipeline.create(dai.node.Camera).build(
    dai.CameraBoardSocket.CAM_B
)

print("Camera node created successfully.")
print("Board Socket:", cam.getBoardSocket())
print()

# Try to get outputs
try:
    outputs = cam.getOutputs()
    print("getOutputs():")
    print(outputs)
except Exception as e:
    print("getOutputs() not available or failed:", e)

print("\n" + "=" * 60)
print("dir(cam) - All available attributes/methods:")
print("=" * 60)
print(dir(cam))