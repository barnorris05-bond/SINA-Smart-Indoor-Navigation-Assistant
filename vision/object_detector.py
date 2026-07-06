"""
vision/object_detector.py
Pure object detector.
Takes RGB frame, returns list of DetectedObject.
No DepthAI or CameraManager knowledge.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class DetectedObject:
    label: str
    confidence: float
    bbox: BoundingBox
    distance: Optional[float] = None


class ObjectDetector:
    """
    Pure vision detector.
    """

    def __init__(self):
        self.loaded = False

    def load_model(self) -> None:
        """Load the detection model."""
        print("[ObjectDetector] Loading model...")
        # Placeholder for real model loading
        self.loaded = True
        print("[ObjectDetector] Model loaded.")

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects in RGB frame.

        Args:
            frame: RGB image as numpy array (H, W, 3)

        Returns:
            List of detected objects.
        """
        if not self.loaded:
            raise RuntimeError(
                "Model has not been loaded. Call load_model() first."
            )

        # Placeholder detections for now
        detections = []

        if frame is not None:
            h, w = frame.shape[:2]
            detections.append(
                DetectedObject(
                    label="person",
                    confidence=0.88,
                    bbox=BoundingBox(x1=w//4, y1=h//4, x2=3*w//4, y2=3*h//4)
                )
            )

        return detections