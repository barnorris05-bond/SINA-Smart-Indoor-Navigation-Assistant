"""
audio/audio_manager.py

Facade coordinating the announcement pipeline:

    NavigationDecision -> Announcer (gate) -> Speaker (async TTS thread)

main.py touches ONLY this class. Clean shutdown is guaranteed by
shutdown(), which stops the TTS thread before the process exits.
"""

import threading
import time
from typing import Optional

from audio.speaker import Speaker
from audio.announcer import Announcer
from utils.logger import logger


class AudioManager:
    """Single entry point for spoken navigation guidance."""

    def __init__(self, enabled: bool = True, clock=time.time) -> None:
        # clock is injectable for deterministic tests; production uses time.time.
        self.enabled = enabled
        self._announcer = Announcer(clock=clock)
        self._speaker = Speaker()
        self._lock = threading.Lock()

    # ======================================================
    # Lifecycle
    # ======================================================

    def start(self) -> None:
        """Start the background TTS thread."""
        if not self.enabled:
            return
        self._speaker.start()

    def shutdown(self) -> None:
        """Stop the TTS thread cleanly (call from main's finally block)."""
        self._speaker.stop()

    # ======================================================
    # Per-frame API (called every loop iteration; MUST be fast)
    # ======================================================

    def update(self, decision) -> None:
        """
        Feed the current NavigationDecision. Non-blocking; internally
        decides whether anything is spoken this frame.
        """
        if not self.enabled:
            return

        with self._lock:
            text = self._announcer.evaluate(decision)

        if text is None:
            return

        # Log suppression decisions implicitly: log the spoken ones only,
        # keeping per-frame logging quiet per project logging rules.
        logger.info(f"Audio | {decision.action.name} | \"{text}\"")
        self._speaker.speak(text)

    # ======================================================
    # Introspection
    # ======================================================

    def is_speaking(self) -> bool:
        return self._speaker.is_talking()

    def reset(self) -> None:
        """Clear announcement state (after outages, mode switches, ...)."""
        with self._lock:
            self._announcer.reset()
