"""
tests/test_minimal_stereo.py
Minimal stereo depth only (no RGB).
"""

import cv2
import depthai as dai

pipeline = dai.Pipeline()

left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

stereo = pipeline.create(dai.node.StereoDepth)

left.requestOutput((640, 480), dai.ImgFrame.Type.GRAY8).link(stereo.left)
right.requestOutput((640, 480), dai.ImgFrame.Type.GRAY8).link(stereo.right)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)

print("Starting minimal stereo pipeline...")
device = dai.Device()
device.startPipeline(pipeline)

q = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

print("Streaming depth. Press 'q' to quit.")

try:
    while True:
        frame = q.tryGet()
        if frame is not None:
            depth = frame.getFrame()
            norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            cv2.imshow("Minimal Depth", colored)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    device.close()
    cv2.destroyAllWindows()