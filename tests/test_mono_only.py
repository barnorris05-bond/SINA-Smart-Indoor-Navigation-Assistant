"""
tests/test_mono_only.py
Test if a simple mono camera pipeline works (no stereo).
"""

import cv2
import depthai as dai

pipeline = dai.Pipeline()

left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
out = left.requestOutput((640, 480), dai.ImgFrame.Type.GRAY8)

device = dai.Device()
device.startPipeline(pipeline)

q = out.createOutputQueue(maxSize=4, blocking=False)

print("Streaming mono left camera. Press 'q' to quit.")

try:
    while True:
        frame = q.tryGet()
        if frame is not None:
            img = frame.getCvFrame()
            cv2.imshow("Mono Left", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    device.close()
    cv2.destroyAllWindows()