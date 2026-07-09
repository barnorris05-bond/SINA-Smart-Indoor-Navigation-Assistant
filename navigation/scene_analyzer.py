"""
navigation/scene_analyzer.py
Extracts regional collections of factual object maps. No policy/threshold logic.
"""

from dataclasses import dataclass, field
from typing import List, Dict
from vision.object_detector import DetectedObject


@dataclass(frozen=True)
class SceneAnalysisReport:
    """Immutable factual breakdown of environmental distributions."""
    center_objects: List[DetectedObject] = field(default_factory=list)
    left_objects: List[DetectedObject] = field(default_factory=list)
    right_objects: List[DetectedObject] = field(default_factory=list)
    closest_distance: float = 0.0
    object_counts: Dict[str, int] = field(default_factory=dict)


class SceneAnalyzer:
    """Compiles raw visual array segments into structural region reports."""

    def analyze(self, detections: List[DetectedObject]) -> SceneAnalysisReport:
        """Categorizes current frame detections purely by spatial distribution facts."""
        counts: Dict[str, int] = {}
        closest_dist = float('inf')
        
        center_list = []
        left_list = []
        right_list = []

        for d in detections:
            label_lower = d.label.lower()
            counts[label_lower] = counts.get(label_lower, 0) + 1
            
            if d.distance is not None and d.distance < closest_dist:
                closest_dist = d.distance

            # Objectively partition lanes without executing policy checks
            if d.region == "CENTER":
                center_list.append(d)
            elif d.region == "LEFT":
                left_list.append(d)
            elif d.region == "RIGHT":
                right_list.append(d)

        return SceneAnalysisReport(
            center_objects=center_list,
            left_objects=left_list,
            right_objects=right_list,
            closest_distance=closest_dist if closest_dist != float('inf') else 0.0,
            object_counts=counts
        )
