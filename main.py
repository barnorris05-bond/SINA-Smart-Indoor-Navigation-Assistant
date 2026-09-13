"""
main.py
System orchestrator showcasing the pipeline: Camera -> Detector -> Manager -> Navigator.
"""

import cv2
import numpy as np
from camera.camera_manager import CameraManager
from vision.yolo_detector import YOLODetector
from vision.detection_manager import DetectionManager
from vision.object_tracker import ObjectTracker
from vision.motion_estimator import MotionEstimator
from navigation.risk_estimator import RiskEstimator
from navigation.distance_navigator import DistanceNavigator
from navigation.temporal_navigator import TemporalNavigator
from depth.provider_factory import create_distance_fusion
from audio import AudioManager
from vision.drawing import draw_boxes, draw_fps, draw_center_marker, draw_regions
from utils.fps import FPS


def main():
    print("Starting Smart Indoor Navigation Assistant...")

    camera = CameraManager()
    detector = YOLODetector()
    manager = DetectionManager()
    # One tracker for the whole application lifetime (Phase 5):
    # persistent track_ids across frames; detection order independent.
    tracker = ObjectTracker()
    # Phase 6: per-track motion classification over fused distances
    # (APPROACHING/STATIONARY/RECEDING/UNKNOWN + closing rate).
    motion = MotionEstimator()
    # Phase 7: per-object risk levels (annotation only for now; Phase 8
    # wires risk into temporal navigation).
    risk = RiskEstimator()
    # Distance-aware policy: falls back to the legacy spatial Navigator
    # whenever no usable distance exists (identical behavior then).
    # Phase 8: temporal stabilization wraps the distance-aware policy.
    # Legacy fallback stays intact inside DistanceNavigator; the temporal
    # layer only smooths de-escalation/action changes and forces STOP on
    # CRITICAL scene risk. One instance for the application lifetime.
    navigator = TemporalNavigator(DistanceNavigator())
    audio = AudioManager()
    # Phase B: mock depth provider (SIMULATED distances) until real
    # stereo is validated; swap happens in depth/provider_factory.py.
    fusion = create_distance_fusion()
    fusion.start()
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
            detections = tracker.update(detections)
            fusion.update()  # provider frame refresh (no-op for mocks)
            fusion.fuse(detections, frame.shape[:2])
            motion.update(detections)  # Phase 6: approach/stationary/receding
            risk.update(detections)    # Phase 7: risk annotation (consumed by Phase 8 layer below)
            decision = navigator.decide(detections)
            audio.update(decision)

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
        audio.shutdown()
        fusion.stop()
        return

    audio.start()
    print("System ready. Press 'q' to quit.")

    try:
        while True:
            frame = camera.get_rgb_frame()
            if frame is not None:
                raw_detections = detector.detect(frame)
                detections = manager.process(raw_detections, frame.shape[1])
                detections = tracker.update(detections)
                fusion.update()  # provider frame refresh (no-op for mocks)
                fusion.fuse(detections, frame.shape[:2])
                motion.update(detections)  # Phase 6: approach/stationary/receding
                risk.update(detections)    # Phase 7: risk annotation (consumed by Phase 8 layer below)
                decision = navigator.decide(detections)
                audio.update(decision)

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
        audio.shutdown()
        fusion.stop()
        camera.stop()
        cv2.destroyAllWindows()
        print("SINA stopped gracefully.")


if __name__ == "__main__":
    main()
