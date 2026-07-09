"""
navigation/navigator.py
Policy decision engine that interprets factual spatial maps using strict semantic configurations.
"""

from typing import List
from vision.object_detector import DetectedObject
from config.navigation import NAVIGATION_OBSTACLES, CRITICAL_PRIORITY_THRESHOLD
from .navigation_types import NavigationAction
from .scene_analyzer import SceneAnalyzer


class NavigationDecision:
    """Structured resolution profile explaining chosen policies."""
    def __init__(self, action: NavigationAction, reason: str, priority: int = 0):
        self.action = action
        self.reason = reason
        self.priority = priority


class Navigator:
    """Translates passive spatial snapshots into prescriptive safety actions."""

    def __init__(self):
        self.analyzer = SceneAnalyzer()

    def _is_hazard(self, obj: DetectedObject) -> bool:
        """Determines if a target object is an active navigational threat using policy limits."""
        return (obj.label.lower() in NAVIGATION_OBSTACLES and 
                obj.priority >= CRITICAL_PRIORITY_THRESHOLD)

    def decide(self, detections: List[DetectedObject]) -> NavigationDecision:
        """Applies dynamic threshold policy criteria to choose safe directional paths."""
        if not detections:
            return NavigationDecision(
                action=NavigationAction.CONTINUE,
                reason="No obstacles detected",
                priority=10
            )

        # Retrieve structural space maps from the SceneAnalyzer
        scene = self.analyzer.analyze(detections)

        # Step 1: Evaluate CENTER path hazards using robust max priority sorting
        if scene.center_objects:
            top_center = max(scene.center_objects, key=lambda d: d.priority)
            
            if self._is_hazard(top_center):
                return NavigationDecision(
                    action=NavigationAction.STOP,
                    reason=f"{top_center.label.capitalize()} blocking CENTER",
                    priority=top_center.priority
                )
            else:
                return NavigationDecision(
                    action=NavigationAction.SLOW_DOWN,
                    reason=f"Low-impact {top_center.label.lower()} in center",
                    priority=top_center.priority
                )

        # Step 2: Clear center lane allows safe evaluation of peripheral threats
        if scene.left_objects:
            top_left = max(scene.left_objects, key=lambda d: d.priority)
            if self._is_hazard(top_left):
                return NavigationDecision(
                    action=NavigationAction.MOVE_RIGHT,
                    reason=f"{top_left.label.capitalize()} hazard on LEFT",
                    priority=top_left.priority
                )

        if scene.right_objects:
            top_right = max(scene.right_objects, key=lambda d: d.priority)
            if self._is_hazard(top_right):
                return NavigationDecision(
                    action=NavigationAction.MOVE_LEFT,
                    reason=f"{top_right.label.capitalize()} hazard on RIGHT",
                    priority=top_right.priority
                )

        return NavigationDecision(
            action=NavigationAction.CONTINUE,
            reason="All operational lanes clear",
            priority=10
        )
