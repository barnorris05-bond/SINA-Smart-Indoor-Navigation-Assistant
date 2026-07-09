"""
main.py
System orchestrator showcasing the pipeline: Camera -> Detector -> Manager -> Navigator.
"""

import cv2
import numpy as np
from camera.camera_manager import CameraManager
from vision.yolo_detector import YOLODetector
from vision.detection_manager import DetectionManager
from navigation.navigator import Navigator
from vision.drawing import draw_boxes, draw_fps, draw_center_marker, draw_regions
from utils.fps import FPS


def main():
    print("Starting Smart Indoor Navigation Assistant...")

    camera = CameraManager()
    detector = YOLODetector()
    manager = DetectionManager()
    navigator = Navigator()   
    fps_counter = FPS()

    detector.load_model()

    try:
        camera.start()
    except Exception as e:
        print(f"Camera startup failed: {e}\nFalling back to synthetic loop...")
        running = True
        while running:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            raw_detections = detector.detect(frame)
            detections = manager.process(raw_detections, frame.shape[1])
            decision = navigator.decide(detections)

            frame = draw_regions(frame)
            frame = draw_boxes(frame, detections)
            frame = draw_fps(frame, fps_counter.update())
            frame = draw_center_marker(frame)

            # Access .name property directly from the Enum object
            cv2.putText(frame, f"Action: {decision.action.name}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("SINA (Camera Offline)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
        return

    print("System ready. Press 'q' to quit.")

    try:
        while True:
            frame = camera.get_rgb_frame()
            if frame is not None:
                raw_detections = detector.detect(frame)
                detections = manager.process(raw_detections, frame.shape[1])
                decision = navigator.decide(detections)

                frame = draw_regions(frame)
                frame = draw_boxes(frame, detections)
                frame = draw_center_marker(frame)
                frame = draw_fps(frame, fps_counter.update())

                # Render policy diagnostics
                cv2.putText(frame, f"Action: {decision.action.name}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Reason: {decision.reason}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("SINA", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("SINA stopped gracefully.")


if __name__ == "__main__":
    main()
