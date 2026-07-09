"""
vision/object_detector.py
Abstract base class for object detectors with tracking telemetry support.
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
    region: str = "CENTER"  # LEFT, CENTER, RIGHT
    width: int = 0
    height: int = 0
    area: int = 0
    priority: int = 0
    distance: Optional[float] = None
    
    # Object Tracking Integration Hook
    track_id: Optional[int] = None  # None indicates tracking inactive or unassigned


class ObjectDetector(ABC):
    """Abstract base class for all object detectors."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """Detect objects in RGB frame."""
        pass
