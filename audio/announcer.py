"""
audio/announcer.py

State-transition gating: decides WHETHER an event should be spoken.

Pure logic, no engine, no threads — fully unit-testable. The Speaker
thread does the actual talking; this class owns announcement policy:

- deduplication of identical consecutive events (state signatures)
- per-action cooldowns (anti-chatter)
- priority escalation (STOP overrides, meaningful escalations bypass cooldown)
- distance escalation: same obstacle significantly closer => re-announce
"""

import time
from typing import Optional

from config.audio import (
    COOLDOWNS,
    COOLDOWN_DEFAULT,
    PRIORITY_MAP,
    PRIORITY_DEFAULT,
    ESCALATION_DELTA,
    ESCALATION_DISTANCE_DELTA,
)
from audio.message_builder import build_message, distance_bucket


class Announcer:
    """Gatekeeper that converts a stream of decisions into sparse announcements."""

    def __init__(self, clock=time.time) -> None:
        # Injectable clock: deterministic tests without sleeping/mocking.
        self._clock = clock
        self.last_signature: Optional[str] = None
        self.last_announce_time: float = 0.0
        self.last_priority: int = 0
        self.last_distance: Optional[float] = None

    # ======================================================
    # Public API
    # ======================================================

    def evaluate(self, decision) -> Optional[str]:
        """
        Evaluate a NavigationDecision; return the text to speak, or None
        when the announcement must be suppressed.
        """
        text, signature = build_message(decision)
        now = self._clock()
        action = decision.action.name
        priority = PRIORITY_MAP.get(action, PRIORITY_DEFAULT)

        # 1) Same event as before: suppress during cooldown, EXCEPT when
        #    the same obstacle came significantly closer (safety escalation).
        if signature == self.last_signature:
            distance = decision.trigger.distance if decision.trigger else None
            closer = (
                distance is not None
                and self.last_distance is not None
                and self.last_distance - distance >= ESCALATION_DISTANCE_DELTA
            )
            if now - self.last_announce_time < COOLDOWNS.get(action, COOLDOWN_DEFAULT) and not closer:
                return None
            # Cooldown expired or obstacle is closing fast: announce.
            self._commit(signature, text, now, priority, decision)
            return text

        # 2) New event. Allow it if the cooldown for its action has
        #    elapsed, OR if it is a meaningful safety escalation.
        cooldown = COOLDOWNS.get(action, COOLDOWN_DEFAULT)
        elapsed = now - self.last_announce_time

        is_escalation = self._is_escalation(decision, priority)

        if elapsed >= cooldown or is_escalation:
            self._commit(signature, text, now, priority, decision)
            return text

        # 3) New but not urgent enough: suppress (anti-chatter).
        return None

    def reset(self) -> None:
        """Forget state (used after outages/recovery)."""
        self.last_signature = None
        self.last_announce_time = 0.0
        self.last_priority = 0
        self.last_distance = None

    # ======================================================
    # Internals
    # ======================================================

    def _is_escalation(self, decision, priority: int) -> bool:
        """A meaningful safety escalation bypasses cooldown."""
        # Priority jump (e.g. SLOW_DOWN -> STOP, CONTINUE -> MOVE_LEFT).
        if priority - self.last_priority >= ESCALATION_DELTA and priority > 1:
            return True

        # Same obstacle came significantly closer.
        distance = decision.trigger.distance if decision.trigger else None
        if (
            distance is not None
            and self.last_distance is not None
            and self.last_distance - distance >= ESCALATION_DISTANCE_DELTA
        ):
            return True

        return False

    def _commit(self, signature, text, now, priority, decision) -> None:
        self.last_signature = signature
        self.last_announce_time = now
        self.last_priority = priority
        self.last_distance = (
            decision.trigger.distance if decision.trigger else None
        )
