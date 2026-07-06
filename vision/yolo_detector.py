"""
vision/yolo_detector.py
YOLO11 detector implementation.
Uses Ultralytics YOLO for real object detection.
"""

from ultralytics import YOLO
import numpy as np
from typing import List
from .object_detector import ObjectDetector, DetectedObject, BoundingBox


class YOLODetector(ObjectDetector):
    """
    YOLO11-based object detector.
    """

    def __init__(self, model_name: str = "yolo11n.pt"):
        self.model_name = model_name
        self.model = None
        self.loaded = False

    def load_model(self) -> None:
        """Load YOLO11 model."""
        print(f"[YOLODetector] Loading {self.model_name}...")
        self.model = YOLO(self.model_name)
        self.loaded = True
        print(f"[YOLODetector] {self.model_name} loaded successfully.")

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects using YOLO11.
        """
        if not self.loaded:
            self.load_model()

        results = self.model(frame, verbose=False)

        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = result.names[cls]

                # Confidence filtering
                if conf < 0.5:   # Filter weak detections
                    continue

                detections.append(
                    DetectedObject(
                        label=label,
                        confidence=conf,
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                    )
                )

        return detections
