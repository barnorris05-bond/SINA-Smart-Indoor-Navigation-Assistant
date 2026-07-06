"""
test_stereo_depth.py

Standalone Stereo Depth test for the OAK-D Lite.
"""

import cv2
import depthai as dai


print("Creating pipeline...")

pipeline = dai.Pipeline()

# ----------------------------
# Left Mono Camera
# ----------------------------

left = pipeline.create(dai.node.MonoCamera)
left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setFps(30)

# ----------------------------
# Right Mono Camera
# ----------------------------

right = pipeline.create(dai.node.MonoCamera)
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setFps(30)

# ----------------------------
# Stereo Depth
# ----------------------------

stereo = pipeline.create(dai.node.StereoDepth)

stereo.setDefaultProfilePreset(
    dai.node.StereoDepth.PresetMode.HIGH_DENSITY
)

# Connect cameras

left.out.link(stereo.left)
right.out.link(stereo.right)

# ----------------------------
# Output Queue
# ----------------------------

depth_output = stereo.depth.createOutputQueue(
    maxSize=4,
    blocking=False
)

print("Starting pipeline...")

pipeline.start()

print("Pipeline running.")
print("Press Q to quit.")

while True:

    depth_frame = depth_output.get().getFrame()

    depth_normalized = cv2.normalize(
        depth_frame,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    depth_normalized = depth_normalized.astype("uint8")

    depth_colored = cv2.applyColorMap(
        depth_normalized,
        cv2.COLORMAP_JET
    )

    cv2.imshow("Stereo Depth", depth_colored)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()