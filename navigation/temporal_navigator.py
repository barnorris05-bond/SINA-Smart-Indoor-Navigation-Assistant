"""
navigation/temporal_navigator.py

TEMPORAL NAVIGATION STABILIZATION (Phase 8).

Answers: "what should SINA do over time?" - the final decision layer
before audio. Composes (never replaces) the existing policies:

    DistanceNavigator (which already falls back to the legacy
    spatial Navigator when no usable distance exists)
            |
            v  candidate NavigationDecision per frame
    TemporalNavigator
            |
            v  stabilized NavigationDecision

Responsibility boundaries preserved (master roadmap):
- ObjectTracker owns identity; DistanceFusion owns distance;
  MotionEstimator owns "what is it doing?"; RiskEstimator remains
  the SOLE authority for risk classification - this layer only READS
  det.risk_level (a string annotation), it never recomputes risk;
  AudioManager owns speech.

POLICY (all thresholds in config/temporal_navigation.py,
DEVELOPMENT/TEST - not calibrated):

1. TOTAL SEVERITY ORDER. Candidate and current states are reduced to
   one severity value:

       severity = max(action_floor, scene_risk_severity)

       action floor: STOP -> 4, SLOW_DOWN / MOVE_* -> 2, CONTINUE -> 1
       scene risk:   max annotated risk over detections
                     (CRITICAL 4, HIGH 3, MEDIUM 2, LOW 1, none 0)

   The action floor fixes the Phase 8 draft flaw where, with no risk
   data, a genuine lateral hazard (CONTINUE -> MOVE_LEFT) was
   "equal severity" and could be suppressed forever. RiskLevel
   (Phase 7 IntEnum) provides the risk ordering - no second severity
   hierarchy is invented. Risk annotation is optional: with no risk
   data the layer still stabilizes on action floors alone, so
   legacy-only operation is preserved.

2. SCENE RISK IS A MAX, NOT A MEAN. One CRITICAL object makes the
   scene CRITICAL - a low-risk object can never dilute a high-risk
   one (tested).

3. SCENE-CRITICAL FORCES STOP (config CRITICAL_RISK_FORCE_STOP).
   Phase 8 is the phase where risk starts to influence navigation:
   when the scene's max annotated risk is CRITICAL, the displayed
   decision becomes STOP even if the distance-band candidate was
   weaker (fast approach just outside the STOP distance band).
   RiskEstimator still owns classification; this is consumption.

4. ESCALATION IS IMMEDIATE. A candidate with severity > current
   severity is displayed the frame it appears. Any STOP candidate is
   additionally force-accepted (CRITICAL_OVERRIDE) - a genuine
   emergency is never delayed by smoothing, including a NEW obstacle
   arriving while already stopped for another.

5. DE-ESCALATION REQUIRES PERSISTENCE. A lower-severity candidate
   must appear on DEESCALATION_CONFIRM_FRAMES consecutive frames
   before the displayed action relaxes. While confirmation is
   pending, the PREVIOUS action is re-emitted (stable audio
   signature) with an auditable "holding ..." reason. A higher or
   equal-severity candidate in between resets the streak.

6. ACTION CHANGES AT EQUAL SEVERITY REQUIRE PERSISTENCE. A switch to
   a different action of equal severity (MOVE_LEFT <-> MOVE_RIGHT
   direction flips, SLOW_DOWN <-> MOVE_* lateral changes) must appear
   on ACTION_CHANGE_CONFIRM_FRAMES consecutive frames. An
   intervening re-confirmation of the current action resets the
   streak, so strictly alternating noisy candidates never switch
   while a genuine persistent change does.

7. STALE COUNTERS. Whenever the scene re-produces the currently
   displayed action at equal severity, ALL pending counters reset -
   "consecutive" means consecutive.

8. TRACK HANDLING. State is scene-level; there is no per-track
   temporal state here, so cross-track leakage is structurally
   impossible. When a tracked object's detection vanishes, its risk
   annotation vanishes from the candidate frame, the scene max risk
   drops, and de-escalation confirmation (rule 5) applies - the
   object is NOT assumed instantly gone, and no dangerous state is
   retained after the scene genuinely clears.

9. PROVENANCE / DATA HONESTY. Decision logic only: never fabricates
   distances, never converts SIMULATED to MEASURED or STALE to
   current, never mutates detections. With distance UNAVAILABLE the
   candidate comes from the legacy spatial fallback inside
   DistanceNavigator and is stabilized exactly like any other.

10. AUDITABILITY. Every displayed decision carries either the
    candidate's own reason or an explicit "holding ..." reason; the
    candidate's trigger is preserved on holds so audio context
    (label, region, distance bucket) stays with the held action.
"""

from typing import List

from vision.object_detector import DetectedObject
from navigation.navigation_types import NavigationAction
from navigation.navigator import NavigationDecision
from navigation.distance_navigator import DistanceNavigator
from config.temporal_navigation import (
    DEESCALATION_CONFIRM_FRAMES,
    ACTION_CHANGE_CONFIRM_FRAMES,
    CRITICAL_OVERRIDE,
    CRITICAL_RISK_FORCE_STOP,
)

# Risk annotation values stamped by the Phase 7 RiskEstimator
# (navigation/risk_estimator.py RiskLevel IntEnum .name values).
_RISK_SEVERITY = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
}

# Intrinsic caution level of each action (see docstring rule 1).
_ACTION_SEVERITY = {
    NavigationAction.STOP: 4,
    NavigationAction.SLOW_DOWN: 2,
    NavigationAction.MOVE_LEFT: 2,
    NavigationAction.MOVE_RIGHT: 2,
    NavigationAction.CONTINUE: 1,
}


class TemporalNavigator:
    """
    Stabilizing wrapper around a candidate navigator (DistanceNavigator).

    Usage: create ONCE for the application lifetime; call decide()
    once per frame with fully annotated detections. reset() is
    provided for scene changes / tests.
    """

    def __init__(
        self,
        candidate_navigator: DistanceNavigator,
        deescalation_confirm_frames: int = DEESCALATION_CONFIRM_FRAMES,
        action_change_confirm_frames: int = ACTION_CHANGE_CONFIRM_FRAMES,
        critical_override: bool = CRITICAL_OVERRIDE,
        critical_risk_force_stop: bool = CRITICAL_RISK_FORCE_STOP,
    ) -> None:
        if deescalation_confirm_frames < 1:
            raise ValueError("deescalation_confirm_frames must be >= 1")
        if action_change_confirm_frames < 1:
            raise ValueError("action_change_confirm_frames must be >= 1")
        self._inner = candidate_navigator
        self._deesc_frames = deescalation_confirm_frames
        self._action_frames = action_change_confirm_frames
        self._critical_override = critical_override
        self._critical_force_stop = critical_risk_force_stop

        self._current = NavigationDecision(
            action=NavigationAction.CONTINUE,
            reason="temporal navigator initialized",
            priority=10,
        )
        self._current_severity = 1

        # Pending de-escalation: consecutive lower-severity candidates.
        self._pending = None                # type: NavigationDecision
        self._pending_severity = 1
        self._pending_count = 0
        # Pending action change: consecutive candidates for the new action.
        self._pending_action = None         # type: NavigationAction
        self._pending_action_count = 0

    # ======================================================
    # Public API
    # ======================================================

    def decide(self, detections: List[DetectedObject]) -> NavigationDecision:
        """
        Produce the stabilized decision for this frame. Non-mutating:
        detections are only read (risk annotation is consumed, never
        recomputed or modified).
        """
        candidate = self._inner.decide(detections)
        scene_risk = self._scene_risk_severity(detections)

        # Rule 3: scene-CRITICAL forces STOP (risk warrants STOP).
        if (
            self._critical_force_stop
            and scene_risk >= _RISK_SEVERITY["CRITICAL"]
            and candidate.action != NavigationAction.STOP
        ):
            source = self._critical_source(detections)
            track_note = (
                f" (track #{source.track_id})"
                if source.track_id is not None else ""
            )
            candidate = NavigationDecision(
                action=NavigationAction.STOP,
                reason=(
                    f"Critical risk: {source.label.lower()}{track_note}"
                ),
                priority=source.priority,
                trigger=source,
            )

        cand_sev = max(
            _ACTION_SEVERITY[candidate.action], scene_risk
        )

        # Rule 4: escalation is immediate.
        if cand_sev > self._current_severity:
            return self._accept(candidate, cand_sev)

        # Rule 4b: critical override - a STOP candidate is never delayed.
        if self._critical_override and candidate.action == NavigationAction.STOP:
            return self._accept(candidate, cand_sev)

        # Rule 5: de-escalation requires consecutive confirmation.
        if cand_sev < self._current_severity:
            if (
                self._pending is not None
                and self._pending.action == candidate.action
                and self._pending_severity == cand_sev
            ):
                self._pending_count += 1
            else:
                self._pending = candidate
                self._pending_severity = cand_sev
                self._pending_count = 1

            if self._pending_count >= self._deesc_frames:
                return self._accept(candidate, cand_sev)
            return self._hold_pending()

        # ---- Equal severity from here on --------------------------

        if candidate.action == self._current.action:
            # Rule 7: the scene re-confirms the displayed decision;
            # every pending counter restarts (consecutive semantics).
            self._reset_pending()
            return self._hold(self._current)

        # Rule 6: action change at equal severity needs confirmation
        # (direction flips AND forward <-> lateral changes).
        if self._pending_action == candidate.action:
            self._pending_action_count += 1
        else:
            self._pending_action = candidate.action
            self._pending_action_count = 1

        if self._pending_action_count >= self._action_frames:
            return self._accept(candidate, cand_sev)
        return self._hold(
            self._current,
            f"holding {self._current.action.name} pending action-change "
            f"confirmation ({self._pending_action_count}/{self._action_frames})",
        )

    def reset(self) -> None:
        """Forget temporal state (scene change / tests)."""
        self._current = NavigationDecision(
            action=NavigationAction.CONTINUE,
            reason="temporal navigator reset",
            priority=10,
        )
        self._current_severity = 1
        self._reset_pending()

    # ======================================================
    # Internals
    # ======================================================

    def _reset_pending(self) -> None:
        self._pending = None
        self._pending_severity = 1
        self._pending_count = 0
        self._pending_action = None
        self._pending_action_count = 0

    @staticmethod
    def _scene_risk_severity(detections: List[DetectedObject]) -> int:
        """
        Max annotated risk over the scene (0 when no risk data exists).
        Max, not min/mean: one CRITICAL object makes the scene
        CRITICAL - a low-risk object can never dilute a high-risk one.
        """
        worst = 0
        for det in detections:
            sev = _RISK_SEVERITY.get(det.risk_level, 0)
            if sev > worst:
                worst = sev
        return worst

    @staticmethod
    def _critical_source(detections: List[DetectedObject]) -> DetectedObject:
        """Highest-priority detection carrying CRITICAL risk."""
        critical = [
            d for d in detections
            if _RISK_SEVERITY.get(d.risk_level, 0)
            >= _RISK_SEVERITY["CRITICAL"]
        ]
        return max(critical, key=lambda d: d.priority)

    def _accept(
        self,
        decision: NavigationDecision,
        severity: int,
    ) -> NavigationDecision:
        """Adopt a decision as the new displayed state."""
        self._current = decision
        self._current_severity = severity
        self._reset_pending()
        return decision

    def _hold_pending(self) -> NavigationDecision:
        """Re-emit the current decision while de-escalation confirms."""
        return self._hold(
            self._current,
            f"holding {self._current.action.name} pending de-escalation "
            f"confirmation ({self._pending_count}/{self._deesc_frames})",
        )

    @staticmethod
    def _hold(
        decision: NavigationDecision,
        reason_override: str = None,
    ) -> NavigationDecision:
        """
        Re-emit the current decision, optionally with an auditable
        hold reason. Trigger is preserved so audio context (label,
        region, distance bucket) stays with the held action.
        """
        if reason_override is None:
            return decision
        return NavigationDecision(
            action=decision.action,
            reason=reason_override,
            priority=decision.priority,
            trigger=decision.trigger,
        )
