"""
vision/detection_manager.py
Enriches raw detections with tracking IDs and external priority policies.
"""

from typing import List
from .object_detector import DetectedObject
from config.navigation import OBJECT_PRIORITIES, MIN_CONFIDENCE


class DetectionManager:
    """Processes raw detections and adds navigation-friendly metadata."""

    def process(
        self,
        detections: List[DetectedObject],
        frame_width: int,
    ) -> List[DetectedObject]:
        """Enrich detections with dynamic spatial regions and shared configuration priorities."""
        enriched = []

        left_boundary = frame_width // 3
        right_boundary = 2 * frame_width // 3

        for det in detections:
            # Fix: Using imported MIN_CONFIDENCE constant
            if det.confidence < MIN_CONFIDENCE:
                continue

            x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            area = width * height

            if center_x < left_boundary:
                region = "LEFT"
            elif center_x < right_boundary:
                region = "CENTER"
            else:
                region = "RIGHT"

            # Fix: Using imported OBJECT_PRIORITIES constant lookup
            priority = OBJECT_PRIORITIES.get(det.label.lower(), 0)

            enriched.append(
                DetectedObject(
                    label=det.label,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    center_x=center_x,
                    center_y=center_y,
                    width=width,
                    height=height,
                    area=area,
                    region=region,
                    priority=priority,
                    distance=det.distance,
                    track_id=det.track_id
                )
            )

        # Sort by priority (highest first)
        enriched.sort(key=lambda d: d.priority, reverse=True)

        return enriched
