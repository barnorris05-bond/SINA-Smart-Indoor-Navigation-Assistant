"""
audio/speaker.py

Asynchronous TTS worker: owns the pyttsx3 engine on a dedicated thread.

Design notes:
- pyttsx3 (SAPI5) is NOT thread-safe across engines; the engine is created,
  used, and shut down entirely on the worker thread.
- speak() is non-blocking: it enqueues text and returns immediately, so the
  camera loop never stalls on speech synthesis or playback.
- Each utterance is spoken with engine.runAndWait() inside the worker.
  That call blocks ONLY the TTS thread (the documented way to drive SAPI5
  reliably); the vision loop is never blocked.
- The worker is a daemon thread with a join-timeout on shutdown, so even a
  wedged SAPI call cannot prevent process exit.
"""

import queue
import threading
from typing import Optional

from config.audio import (
    TTS_RATE,
    TTS_VOLUME,
    SPEECH_QUEUE_SIZE,
    SPEAKER_SHUTDOWN_TIMEOUT,
)


class Speaker:
    """Non-blocking speech queue backed by a single dedicated TTS thread."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue(
            maxsize=SPEECH_QUEUE_SIZE
        )
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._talking = threading.Event()   # True while audio is playing

    # ======================================================
    # Lifecycle
    # ======================================================

    def start(self) -> None:
        """Spawn the background TTS worker thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="SINA-Speaker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = SPEAKER_SHUTDOWN_TIMEOUT) -> None:
        """Request shutdown and wait for the worker to finish cleanly."""
        self._stop_event.set()
        # Unblock the worker if it is waiting on an empty queue.
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    # ======================================================
    # Public API (main thread)
    # ======================================================

    def speak(self, text: str) -> bool:
        """
        Enqueue text for speech. Returns False if the queue is full
        (message dropped rather than blocking the vision loop).
        """
        if not text:
            return False
        try:
            self._queue.put_nowait(text)
            return True
        except queue.Full:
            return False

    def is_talking(self) -> bool:
        """True while an utterance is currently being spoken."""
        return self._talking.is_set()

    # ======================================================
    # Worker internals (TTS thread only)
    # ======================================================

    def _run(self) -> None:
        """Worker loop: create engine once, drain queue until stopped."""
        import pyttsx3  # Imported here: engine must live on this thread.

        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_RATE)
        engine.setProperty("volume", TTS_VOLUME)

        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if text is None:  # Shutdown sentinel
                break

            self._speak_one(engine, text)

        try:
            engine.stop()
        except Exception:
            pass

    def _speak_one(self, engine, text: str) -> None:
        """Speak a single utterance, blocking only this worker thread."""
        self._talking.set()
        try:
            engine.say(text)
            # runAndWait drives SAPI5's COM loop until the utterance queue
            # drains. Blocks this thread only — by design.
            engine.runAndWait()
        except Exception as error:
            # Speech failure must never crash the vision loop.
            print(f"[Speaker] TTS error: {error}")
        finally:
            self._talking.clear()
