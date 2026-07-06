"""
main.py
Entry point for Smart Indoor Navigation Assistant (SINA).
"""

import cv2
import numpy as np
from camera.camera_manager import CameraManager
from vision.yolo_detector import YOLODetector
from vision.drawing import draw_boxes, draw_fps, draw_center_marker
from utils.fps import FPS


def main():
    print("Starting Smart Indoor Navigation Assistant...")

    camera = CameraManager()
    detector = YOLODetector()   # Real YOLO detector
    fps_counter = FPS()

    # Load model
    detector.load_model()

    try:
        camera.start()
    except Exception as e:
        print(f"Camera startup failed: {e}")
        print("Continuing with dummy frames...")
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

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Optional: Reset camera
                camera.reset()

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("SINA stopped.")


if __name__ == "__main__":
    main()