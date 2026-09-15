"""
navigation/risk_estimator.py

RISK ESTIMATION (Phase 7).

Answers, per object: "how dangerous is the situation?" - deliberately
separate from MotionEstimator ("what is the object doing?"),
DistanceNavigator ("what should SINA do?") and AudioManager ("what
should SINA tell the user?"). No navigation logic lives here; no risk
logic lives in the tracker/fusion/motion layers.

Consumes (per detection, all produced by the existing layers):

    label, priority, region, confidence,
    distance, distance_provenance,
    motion_state, closing_rate_mps, motion_provenance

Produces per object:

    RiskLevel   LOW / MEDIUM / HIGH / CRITICAL
    RiskAssessment with explicit human-readable reasons and,
    when trustworthy, an estimated time-to-collision (TTC).

Design (Phase 7 requirements):

- TRANSPARENT LADDER, not a magic formula. Every assessment carries
  the reasons that fired, so any decision can be audited.
- NOT DISTANCE-ONLY: motion state and closing rate escalate risk
  (an APPROACHING person outranks a STATIONARY one at the same
  distance); priority escalates risk (person vs book).
- TTC computed ONLY when trustworthy: motion_state must be
  APPROACHING (hysteresis-confirmed by Phase 6) with closing rate
  >= TTC_MIN_CLOSING_RATE_MPS. TTC is an ESTIMATE, never a guaranteed
  collision time; STALE/UNAVAILABLE/UNKNOWN data yields TTC=None.
- CONSERVATIVE UNKNOWN: without usable distance, high-priority
  obstacles are rated MEDIUM (never silently LOW). Distance provenance
  must be MEASURED (or SIMULATED while ALLOW_SIMULATED_FOR_NAVIGATION)
  exactly like DistanceNavigator's usability rule.
- ANTI-FLAP: risk may rise instantly, but only drops after
  RISK_DROP_CONFIRM_FRAMES consecutive assessments at the lower level
  (per track). One noisy frame cannot turn CRITICAL into LOW.
- CONFIDENCE DOES NOT LOWER RISK (conservative): a shaky person
  detection at 0.9 m stays CRITICAL. Low confidence only ADDS an
  advisory reason ("detection uncertain").
- DEVELOPMENT/TEST thresholds: imported from config/navigation_distance.py,
  config/navigation.py and config/risk.py - none are safety-calibrated.

Integration: purely additive annotation. In main.py it runs after
motion.update() and before navigation, but nothing consumes it yet
(Phase 8 wires risk into temporal navigation). Navigation and audio
semantics are unchanged this phase.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

from vision.object_detector import DetectedObject
from config.navigation_distance import (
    CRITICAL_DISTANCE_M,
    WARNING_DISTANCE_M,
    FAR_DISTANCE_M,
)
from config.navigation import CRITICAL_PRIORITY_THRESHOLD
from config.depth_fusion import ALLOW_SIMULATED_FOR_NAVIGATION
from config.risk import (
    TTC_CRITICAL_S,
    TTC_WARNING_S,
    TTC_MIN_CLOSING_RATE_MPS,
    RISK_DROP_CONFIRM_FRAMES,
)


class RiskLevel(IntEnum):
    """Ordered severity - comparable with <, >, min, max."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class RiskAssessment:
    """Per-object risk result with auditable reasons."""

    level: RiskLevel
    reasons: List[str] = field(default_factory=list)
    ttc_s: Optional[float] = None       # estimated TTC (not a guarantee)
    provenance: str = "UNAVAILABLE"     # MEASURED | SIMULATED | UNAVAILABLE


def distance_usable(det: DetectedObject) -> bool:
    """Mirror of DistanceNavigator's usability rule (single policy)."""
    if det.distance is None:
        return False
    if det.distance_provenance == "MEASURED":
        return True
    if det.distance_provenance == "SIMULATED":
        return ALLOW_SIMULATED_FOR_NAVIGATION
    return False


class RiskEstimator:
    """
    Per-object risk classifier over the existing perception layers.

    Usage: create ONCE for the application lifetime; call assess()
    (or update()) once per frame with tracked/fused/motion-stamped
    detections. Anti-flap state is keyed by track_id; objects without
    a track are assessed statelessly (no hysteresis memory).
    """

    def __init__(self) -> None:
        # track_id -> (displayed level, consecutive frames at that level)
        self._level_streak: dict = {}
        # track_id -> consecutive assessments proposing a LOWER level
        # (reset whenever an assessment proposes previous or higher).
        self._lower_streak: dict = {}

    # ======================================================
    # Public API
    # ======================================================

    def update(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        """Assess every detection and stamp additive risk fields."""
        for det in detections:
            assessment = self.assess(det)
            det.risk_level = assessment.level.name
            det.risk_reasons = list(assessment.reasons)
            det.risk_ttc_s = assessment.ttc_s
            det.risk_provenance = assessment.provenance
        return detections

    def assess(self, det: DetectedObject) -> RiskAssessment:
        """
        Assess one object. Stateless callers may use this directly;
        update() adds per-track anti-flap hysteresis on top.
        """
        usable = distance_usable(det)
        reasons: List[str] = []
        ttc: Optional[float] = None

        if usable:
            if det.distance_provenance == "MEASURED":
                provenance = "MEASURED"
            else:
                provenance = "SIMULATED"

            # Trustworthy TTC (see module docstring for the gate):
            ttc = self._trustworthy_ttc(det)
            if ttc is not None:
                reasons.append(f"TTC ~{ttc:.1f}s (closing {det.closing_rate_mps:.2f}m/s)")

            level = self._ladder(det, ttc, reasons)
        else:
            # No usable distance: rate by what IS known, conservatively.
            provenance = "UNAVAILABLE"
            if det.priority >= CRITICAL_PRIORITY_THRESHOLD:
                level = RiskLevel.MEDIUM
                reasons.append(
                    f"no usable distance; high-priority {det.label.lower()} "
                    f"(priority {det.priority}) treated conservatively"
                )
            else:
                level = RiskLevel.LOW
                reasons.append("no usable distance; low-priority object")

        if det.motion_state == "UNKNOWN" and usable:
            reasons.append("motion unknown (insufficient history)")

        if det.confidence < 0.70:
            # Advisory only - never lowers the level (conservative).
            reasons.append(f"detection uncertain (conf {det.confidence:.2f})")

        level = self._apply_antiflap(det, level)
        return RiskAssessment(level, reasons, ttc, provenance)

    def reset(self) -> None:
        """Clear anti-flap memory (tracker reset / scene change)."""
        self._level_streak.clear()
        self._lower_streak.clear()

    # ======================================================
    # Rule ladder
    # ======================================================

    def _ladder(
        self,
        det: DetectedObject,
        ttc: Optional[float],
        reasons: List[str],
    ) -> RiskLevel:
        """
        Distance/motion/priority ladder. Thresholds are shared with
        navigation (config/navigation_distance.py) - see config/risk.py.
        """
        d = det.distance
        high_priority = det.priority >= CRITICAL_PRIORITY_THRESHOLD
        approaching = det.motion_state == "APPROACHING"

        # ---- CRITICAL ------------------------------------------------
        if d <= CRITICAL_DISTANCE_M:
            reasons.append(
                f"distance {d:.1f}m <= critical {CRITICAL_DISTANCE_M:.1f}m")
            return RiskLevel.CRITICAL
        if ttc is not None and ttc <= TTC_CRITICAL_S:
            reasons.append(f"TTC {ttc:.1f}s <= {TTC_CRITICAL_S:.1f}s")
            return RiskLevel.CRITICAL

        # ---- HIGH ----------------------------------------------------
        if ttc is not None and ttc <= TTC_WARNING_S:
            reasons.append(f"TTC {ttc:.1f}s <= {TTC_WARNING_S:.1f}s")
            return RiskLevel.HIGH
        if d < WARNING_DISTANCE_M and (high_priority or approaching):
            qualifier = " + ".join(
                q for q in ("high-priority" if high_priority else "",
                            "approaching" if approaching else "") if q
            )
            reasons.append(
                f"distance {d:.1f}m < warning {WARNING_DISTANCE_M:.1f}m "
                f"+ {qualifier}")
            return RiskLevel.HIGH
        if high_priority and approaching and d < FAR_DISTANCE_M:
            reasons.append(
                f"high-priority {det.label.lower()} approaching at {d:.1f}m")
            return RiskLevel.HIGH

        # ---- MEDIUM --------------------------------------------------
        if d < WARNING_DISTANCE_M:
            reasons.append(
                f"distance {d:.1f}m < warning {WARNING_DISTANCE_M:.1f}m")
            return RiskLevel.MEDIUM
        if high_priority and d < FAR_DISTANCE_M:
            reasons.append(
                f"high-priority {det.label.lower()} at {d:.1f}m")
            return RiskLevel.MEDIUM

        # ---- LOW -----------------------------------------------------
        reasons.append(f"far ({d:.1f}m >= {FAR_DISTANCE_M:.1f}m), low priority")
        return RiskLevel.LOW

    @staticmethod
    def _trustworthy_ttc(det: DetectedObject) -> Optional[float]:
        """
        TTC ~= distance / closing_rate, ONLY when the motion layer's
        state is hysteresis-confirmed APPROACHING and the rate is at
        least TTC_MIN_CLOSING_RATE_MPS. Everything else -> None
        (no meaningless TTC from stale/unknown/receding data).
        """
        if det.motion_state != "APPROACHING":
            return None
        if det.closing_rate_mps is None or det.closing_rate_mps <= 0:
            return None
        if det.closing_rate_mps < TTC_MIN_CLOSING_RATE_MPS:
            return None
        if det.distance is None or det.distance <= 0:
            return None
        return det.distance / det.closing_rate_mps

    # ======================================================
    # Anti-flap
    # ======================================================

    def _apply_antiflap(self, det: DetectedObject, proposed: RiskLevel) -> RiskLevel:
        """
        Escalate instantly; de-escalate only after the proposed lower
        level has been seen RISK_DROP_CONFIRM_FRAMES times in a row.
        Tracks without an id are assessed statelessly.
        """
        key = det.track_id
        if key is None:
            return proposed

        previous, streak = self._level_streak.get(key, (None, 0))

        if previous is None or proposed > previous:
            # First sight or escalation: immediate. Any pending
            # de-escalation evidence is discarded.
            self._lower_streak.pop(key, None)
            self._level_streak[key] = (proposed, 1)
            return proposed

        if proposed == previous:
            self._lower_streak.pop(key, None)
            self._level_streak[key] = (previous, streak + 1)
            return previous

        # proposed < previous: count CONSECUTIVE lower proposals; the
        # displayed level holds until RISK_DROP_CONFIRM_FRAMES of them
        # accumulate. A single higher proposal resets the count.
        lower_count = self._lower_streak.get(key, 0) + 1
        self._lower_streak[key] = lower_count
        if lower_count >= RISK_DROP_CONFIRM_FRAMES:
            self._lower_streak.pop(key, None)
            self._level_streak[key] = (proposed, 1)
            return proposed
        return previous
