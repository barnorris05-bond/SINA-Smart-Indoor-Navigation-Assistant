"""
navigation/distance_navigator.py

Distance-aware navigation policy (Phase D).

Relationship to the legacy Navigator (navigation/navigator.py):

- ADDITIVE, not a replacement. decide() uses distances ONLY from
  usable measurements (MEASURED, or SIMULATED while mock depth is
  enabled). If no object in the scene carries a usable distance, the
  call is delegated to the legacy spatial Navigator unchanged — so the
  existing Sprint 4 behavior is preserved byte-for-byte when depth is
  absent.
- When usable distances exist, this policy takes precedence:

      any object <= CRITICAL_DISTANCE_M            -> STOP
      center object <  WARNING_DISTANCE_M          -> SLOW_DOWN
      lateral object <  WARNING_DISTANCE_M         -> move away
      all distances >= FAR_DISTANCE_M              -> CONTINUE

Thresholds come from config/navigation_distance.py and are
DEVELOPMENT/TEST values pending real-depth calibration (Phase E/F).
"""

from typing import List, Optional

from vision.object_detector import DetectedObject
from navigation.navigation_types import NavigationAction
from navigation.navigator import Navigator, NavigationDecision
from config.navigation_distance import (
    CRITICAL_DISTANCE_M,
    WARNING_DISTANCE_M,
    FAR_DISTANCE_M,
)
from config.depth_fusion import ALLOW_SIMULATED_FOR_NAVIGATION


class DistanceNavigator:
    """Decides navigation actions using distances when they exist."""

    def __init__(self, spatial_fallback: Optional[Navigator] = None) -> None:
        # Legacy policy is reused as fallback (no behavior change there).
        self._fallback = spatial_fallback or Navigator()

    # ------------------------------------------------------

    def decide(self, detections: List[DetectedObject]) -> NavigationDecision:
        if not detections:
            return self._fallback.decide(detections)

        usable = [
            d for d in detections
            if self._is_usable(d)
        ]
        if not usable:
            # No distance information this frame -> legacy spatial policy.
            return self._fallback.decide(detections)

        # Step 1: anything critically close, any region -> STOP.
        critical = [d for d in usable if d.distance <= CRITICAL_DISTANCE_M]
        if critical:
            target = self._most_urgent(critical)
            return NavigationDecision(
                action=NavigationAction.STOP,
                reason=(
                    f"{target.label.capitalize()} at "
                    f"{target.distance:.1f}m ({target.distance_provenance})"
                ),
                priority=target.priority,
                trigger=target,
            )

        # Step 2: center obstacles within the warning band -> SLOW_DOWN.
        # Band is inclusive (<=) so the reference scenario "chair at
        # 2.5m -> SLOW_DOWN" holds at the current dev threshold.
        center = [d for d in usable if d.region == "CENTER"]
        if center:
            target = self._most_urgent(
                [d for d in center if d.distance <= WARNING_DISTANCE_M]
            )
            if target is not None:
                return NavigationDecision(
                    action=NavigationAction.SLOW_DOWN,
                    reason=(
                        f"{target.label.capitalize()} ahead at "
                        f"{target.distance:.1f}m ({target.distance_provenance})"
                    ),
                    priority=target.priority,
                    trigger=target,
                )

        # Step 3: lateral obstacles within the warning band -> move away.
        left = [
            d for d in usable
            if d.region == "LEFT" and d.distance <= WARNING_DISTANCE_M
        ]
        if left:
            target = self._most_urgent(left)
            return NavigationDecision(
                action=NavigationAction.MOVE_RIGHT,
                reason=(
                    f"{target.label.capitalize()} on LEFT at "
                    f"{target.distance:.1f}m ({target.distance_provenance})"
                ),
                priority=target.priority,
                trigger=target,
            )

        right = [
            d for d in usable
            if d.region == "RIGHT" and d.distance <= WARNING_DISTANCE_M
        ]
        if right:
            target = self._most_urgent(right)
            return NavigationDecision(
                action=NavigationAction.MOVE_LEFT,
                reason=(
                    f"{target.label.capitalize()} on RIGHT at "
                    f"{target.distance:.1f}m ({target.distance_provenance})"
                ),
                priority=target.priority,
                trigger=target,
            )

        # Step 4: everything known is far enough -> continue.
        closest = min(usable, key=lambda d: d.distance)
        return NavigationDecision(
            action=NavigationAction.CONTINUE,
            reason=(
                f"Closest obstacle {closest.label.lower()} at "
                f"{closest.distance:.1f}m (>= {FAR_DISTANCE_M:.1f}m) "
                f"({closest.distance_provenance})"
            ),
            priority=10,
        )

    # ------------------------------------------------------

    @staticmethod
    def _is_usable(det: DetectedObject) -> bool:
        """Usable = has a distance with acceptable provenance."""
        if det.distance is None:
            return False
        if det.distance_provenance == "MEASURED":
            return True
        if det.distance_provenance == "SIMULATED":
            # Development/test flag: simulated values drive navigation
            # ONLY while mock depth mode is enabled.
            return ALLOW_SIMULATED_FOR_NAVIGATION
        return False  # UNAVAILABLE / STALE never drive decisions

    @staticmethod
    def _most_urgent(candidates: List[DetectedObject]) -> Optional[DetectedObject]:
        """Closest first; ties broken by priority (higher wins)."""
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda d: (d.distance, -d.priority),
        )
