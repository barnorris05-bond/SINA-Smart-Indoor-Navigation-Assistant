"""
tests/test_official_stereo.py
Standalone official-style stereo depth test (DepthAI v3).
No CameraManager. No builder. Pure SDK.
"""

import cv2
import depthai as dai


def main():
    print("Creating official-style stereo pipeline...")

    pipeline = dai.Pipeline()

    # Left and Right Mono Cameras
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # Stereo Depth
    stereo = pipeline.create(dai.node.StereoDepth)
    left.requestOutput((1280, 720), dai.ImgFrame.Type.GRAY8).link(stereo.left)
    right.requestOutput((1280, 720), dai.ImgFrame.Type.GRAY8).link(stereo.right)

    # Use a preset that exists in your SDK
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    print("Starting device...")
    device = dai.Device()
    device.startPipeline(pipeline)

    print("Device started. Creating queue...")
    q = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

    print("Streaming depth. Press 'q' to quit.")

    try:
        while True:
            frame = q.tryGet()
            if frame is not None:
                depth_data = frame.getFrame()
                depth_norm = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                cv2.imshow("Official Depth Preview", depth_colored)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        device.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()