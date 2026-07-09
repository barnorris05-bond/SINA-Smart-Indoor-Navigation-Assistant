"""
vision/drawing.py
Rendering suite featuring high-contrast semi-transparent alpha overlays and tracking telemetry.
"""

import cv2
import numpy as np
from typing import List
from .object_detector import DetectedObject


def draw_boxes(frame: np.ndarray, detections: List[DetectedObject]) -> np.ndarray:
    """Renders bounding contours matched with enclosed stacked metadata rectangles."""
    for det in detections:
        x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2
        
        # Bounding border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label injection logic featuring persistent tracking IDs
        label_text = f"{det.label.capitalize()}"
        if det.track_id is not None:
            label_text = f"#{det.track_id} {label_text}"

        # Data collection rows
        text_lines = [
            label_text,
            f"{int(det.confidence * 100)}%",
            f"{det.region}"
        ]
        if det.distance is not None:
            text_lines.append(f"{det.distance:.2f} m")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        line_height = 16
        
        max_w = 0
        for line in text_lines:
            (w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            if w > max_w:
                max_w = w
                
        box_w = max_w + 12
        box_h = (len(text_lines) * line_height) + 8
        
        rx1 = max(x1, 4)
        ry1 = max(y1 - box_h - 4, 4)
        rx2 = rx1 + box_w
        ry2 = ry1 + box_h

        # Overlay a semi-transparent black rectangle background box
        overlay = frame.copy()
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw inner text elements on top of the alpha background container
        for idx, line in enumerate(text_lines):
            tx = rx1 + 6
            ty = ry1 + 14 + (idx * line_height)
            cv2.putText(frame, line, (tx, ty), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    return frame


def draw_regions(frame: np.ndarray) -> np.ndarray:
    """Draws thin partition guide marks splitting frame bounds into dynamic thirds."""
    height, width = frame.shape[:2]
    left_boundary = width // 3
    right_boundary = 2 * width // 3
    color = (80, 80, 80)
    
    for y in range(0, height, 20):
        cv2.line(frame, (left_boundary, y), (left_boundary, y + 10), color, 1)
        cv2.line(frame, (right_boundary, y), (right_boundary, y + 10), color, 1)
        
    cv2.putText(frame, "LEFT", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, "CENTER", (left_boundary + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, "RIGHT", (right_boundary + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def draw_center_marker(frame: np.ndarray) -> np.ndarray:
    """Plots crosshair target tracking vectors directly center mass."""
    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2
    cv2.line(frame, (cx - 12, cy), (cx + 12, cy), (255, 0, 0), 2)
    cv2.line(frame, (cx, cy - 12), (cx, cy + 12), (255, 0, 0), 2)
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Renders real-time telemetry framing frequencies."""
    text = f"FPS: {fps:.1f}"
    # Safeguard indexing resolution errors via dynamic bounding extraction 
    cv2.putText(frame, text, (frame.shape[1] - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    return frame
