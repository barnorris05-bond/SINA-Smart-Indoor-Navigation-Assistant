"""
vision/object_detector.py
Abstract base class for object detectors.
"""

from abc import ABC, abstractmethod
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

    # Navigation metadata
    center_x: int = 0
    center_y: int = 0
    region: str = "CENTER"   # LEFT, CENTER, RIGHT
    area: int = 0
    distance: Optional[float] = None


class ObjectDetector(ABC):
    """
    Abstract base class for all object detectors.
    """

    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects in RGB frame.
        """
        pass