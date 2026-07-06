"""
main.py
Entry point for Smart Indoor Navigation Assistant (SINA).
"""

import cv2
import numpy as np
from camera.camera_manager import CameraManager
from vision.object_detector import ObjectDetector
from vision.drawing import draw_boxes, draw_fps, draw_center_marker
from utils.fps import FPS


def main():
    print("Starting Smart Indoor Navigation Assistant...")

    # Initialize modules
    camera = CameraManager()
    detector = ObjectDetector()
    fps_counter = FPS()

    # Load model
    detector.load_model()

    try:
        camera.start()
    except Exception as e:
        print(f"Camera startup failed: {e}")
        print("Continuing with dummy frames for development...")
        # Dummy mode
        running = True
        while running:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            detections = detector.detect(frame)
            frame = draw_boxes(frame, detections)
            frame = draw_fps(frame, fps_counter.update())
            frame = draw_center_marker(frame)
            cv2.imshow("SINA (Camera Offline)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
        return

    # Normal mode
    print("System ready. Press 'q' to quit.")

    try:
        while True:
            frame = camera.get_rgb_frame()
            if frame is not None:
                detections = detector.detect(frame)
                frame = draw_boxes(frame, detections)
                frame = draw_center_marker(frame)
                frame = draw_fps(frame, fps_counter.update())
                cv2.imshow("SINA", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("SINA stopped.")


if __name__ == "__main__":
    main()