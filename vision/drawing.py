"""
vision/drawing.py
Visualization utilities for detections.
"""

import cv2
import numpy as np
from typing import List

# Import from same package
from .object_detector import DetectedObject


def draw_boxes(frame: np.ndarray, detections: List[DetectedObject]) -> np.ndarray:
    """
    Draw bounding boxes and labels on the frame.
    """
    for det in detections:
        x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label
        label = f"{det.label.capitalize()} ({int(det.confidence*100)}%)"

        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS in top-left corner."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


def draw_center_marker(frame: np.ndarray) -> np.ndarray:
    """Draw center crosshair."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 1)
    return frame
