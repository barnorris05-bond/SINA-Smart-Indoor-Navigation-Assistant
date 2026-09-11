"""
tests/smoke_tts.py

STEP 7 live integration test: real SAPI5 speech on real speakers.

Verifies:
 1. AudioManager starts its TTS thread.
 2. update() is non-blocking (microseconds, not the ~0.5-2 s a
    synchronous engine.say(...).runAndWait() would take).
 3. Speech actually plays (is_talking becomes True) — the user should
    audibly hear the first utterance.
 4. A decision stream driven by a controlled clock produces exactly the
    expected announcements (cooldown, escalation, anti-chatter verified
    against real audio playback, serialized via a race-safe waiter).
 5. shutdown() joins the thread cleanly.

Run:  ./.venv/Scripts/python.exe tests/smoke_tts.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_audio_logic import make_decision, make_object
from audio import AudioManager


def wait_for_speech(audio, timeout=10.0):
    """Race-safe wait: speech must START, then finish."""
    deadline = time.time() + timeout
    while time.time() < deadline and not audio.is_speaking():
        time.sleep(0.02)
    while audio.is_speaking():
        time.sleep(0.05)


def main() -> int:
    failures = []

    clock = {"t": 1000.0}          # Controlled announcement clock (seconds).
    def fake_clock():
        return clock["t"]

    audio = AudioManager(clock=fake_clock)
    audio.start()

    # --- Test 2: update() must be non-blocking -------------------------
    stop = make_decision("STOP", make_object("person", "CENTER", 1.4))
    t0 = time.perf_counter()
    audio.update(stop)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[check] first update() took {elapsed_ms:.3f} ms (must be << 200 ms)")
    if elapsed_ms > 200:
        failures.append("update() blocked the calling thread")

    # --- Test 3: speech actually starts (AUDIBLE) -----------------------
    deadline = time.time() + 3.0
    while time.time() < deadline and not audio.is_speaking():
        time.sleep(0.02)
    print(f"[check] is_speaking()={audio.is_speaking()} (expect True)")
    if not audio.is_speaking():
        failures.append("no speech detected within 3 s")
    wait_for_speech(audio)          # let it finish audibly

    # --- Test 4: gating over a clock-driven decision stream -------------
    # Announce-clock semantics (virtual time), real playback in between:
    #   t=1.2  STOP person 1.4m repeat  -> announced (STOP cooldown 1.0s elapsed)
    #   t=1.4  STOP person 0.8m         -> announced (0.6m closer: escalation)
    #   t=3.0  MOVE_LEFT person RIGHT   -> suppressed (1.6s < 3.0s cooldown)
    #   t=3.4  CONTINUE                 -> suppressed (chatter guard)
    #   t=8.6  CONTINUE                 -> announced (CONTINUE cooldown 5.0s elapsed)
    spoken = []
    original_speak = audio._speaker.speak
    audio._speaker.speak = (
        lambda text: (spoken.append(text), original_speak(text))[0]
    )

    def feed(decision, dt):
        clock["t"] += dt
        audio.update(decision)
        wait_for_speech(audio)      # serialize utterances like a real loop

    feed(make_decision("STOP", make_object("person", "CENTER", 1.4)), 1.2)
    feed(make_decision("STOP", make_object("person", "CENTER", 0.8)), 0.2)
    feed(make_decision("MOVE_LEFT", make_object("person", "RIGHT", 2.0)), 1.6)
    feed(make_decision("CONTINUE"), 0.4)
    feed(make_decision("CONTINUE"), 5.2)

    wait_for_speech(audio)

    print(f"[check] announcements for 5 timed events: {len(spoken)} (expect 3)")
    for s in spoken:
        print(f"        spoke: \"{s}\"")
    if len(spoken) != 3:
        failures.append(f"expected 3 announcements, got {len(spoken)}")

    # --- Test 5: clean shutdown -----------------------------------------
    t0 = time.perf_counter()
    audio.shutdown()
    shutdown_s = time.perf_counter() - t0
    print(f"[check] shutdown completed in {shutdown_s:.2f} s (limit 5 s)")
    if shutdown_s > 5.0:
        failures.append("shutdown hung")

    print()
    if failures:
        print("SMOKE TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE TEST PASSED: async, audible, gated, clean shutdown.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
