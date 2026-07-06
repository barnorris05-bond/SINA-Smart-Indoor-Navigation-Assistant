"""
tests/test_official_stereo_v3.py
Official-style stereo depth test (non-context manager).
"""

import cv2
import depthai as dai

pipeline = dai.Pipeline()

left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

stereo = pipeline.create(dai.node.StereoDepth)

left.requestOutput((1280, 720), dai.ImgFrame.Type.GRAY8).link(stereo.left)
right.requestOutput((1280, 720), dai.ImgFrame.Type.GRAY8).link(stereo.right)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

print("Starting device...")
device = dai.Device()
device.startPipeline(pipeline)

print("Streaming depth. Press 'q' to quit.")

q = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

try:
    while True:
        frame = q.get()
        depth = frame.getFrame()
        norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        cv2.imshow("Depth", colored)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    device.close()
    cv2.destroyAllWindows()