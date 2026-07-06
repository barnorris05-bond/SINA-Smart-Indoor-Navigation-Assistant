"""
tests/test_depth_only.py
Minimal depth-only test.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from camera.stereo_pipeline import StereoPipelineBuilder
import cv2
import depthai as dai


def main():
    print("Creating minimal depth pipeline...")
    pipeline, depth_out = StereoPipelineBuilder.create_depth_only()

    print("Starting device...")
    device = dai.Device()
    device.startPipeline(pipeline)

    print("Device started. Creating queue...")
    depth_queue = depth_out.createOutputQueue(maxSize=4, blocking=False)

    print("Streaming depth. Press 'q' to quit.")

    try:
        while True:
            frame = depth_queue.tryGet()
            if frame is not None:
                depth_data = frame.getFrame()
                depth_norm = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                cv2.imshow("Depth Only", depth_colored)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        device.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()