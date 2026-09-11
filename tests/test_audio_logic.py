"""
tests/test_audio_logic.py

Phase 1 logic tests — no TTS engine, no camera, no sleeping.

Uses a controllable fake clock so cooldown behavior is deterministic.
Covers: NavigationDecision.trigger, dedup, cooldowns, priority escalation,
distance escalation, and the Speaker queue contract.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.object_detector import DetectedObject, BoundingBox
from navigation.navigator import NavigationDecision
from navigation.navigation_types import NavigationAction
from audio.announcer import Announcer
from audio.message_builder import build_message, distance_bucket, round_distance
from audio.speaker import Speaker


def make_object(label="person", region="CENTER", distance=None, priority=100):
    """Build a DetectedObject positioned in the requested region (frame width 640)."""
    if region == "CENTER":
        x1, x2 = 240, 320
    elif region == "LEFT":
        x1, x2 = 20, 100
    else:  # RIGHT
        x1, x2 = 540, 620
    return DetectedObject(
        label=label,
        confidence=0.9,
        bbox=BoundingBox(x1=x1, y1=100, x2=x2, y2=300),
        center_x=(x1 + x2) // 2,
        center_y=200,
        region=region,
        priority=priority,
        distance=distance,
    )


def make_decision(action, trigger=None):
    reasons = {
        "STOP": "Person blocking CENTER",
        "SLOW_DOWN": "Low-impact bottle in center",
        "MOVE_LEFT": "Person hazard on RIGHT",
        "MOVE_RIGHT": "Chair hazard on LEFT",
        "CONTINUE": "All operational lanes clear",
    }
    return NavigationDecision(
        action=NavigationAction[action],
        reason=reasons[action],
        priority=trigger.priority if trigger else 10,
        trigger=trigger,
    )


class FakeClock:
    """Deterministic time source advanced explicitly by tests."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class TestNavigationDecisionTrigger(unittest.TestCase):
    """Regression guard: trigger is additive, defaults must keep old calls valid."""

    def test_decision_defaults_to_no_trigger(self):
        d = NavigationDecision(NavigationAction.CONTINUE, "clear")
        self.assertIsNone(d.trigger)
        self.assertEqual(d.action, NavigationAction.CONTINUE)

    def test_stop_decision_carries_trigger(self):
        person = make_object("person", "CENTER", distance=1.4)
        d = make_decision("STOP", person)
        self.assertIs(d.trigger, person)


class TestMessageBuilder(unittest.TestCase):
    def test_stop_message_with_distance(self):
        decision = make_decision("STOP", make_object("person", distance=1.4))
        text, sig = build_message(decision)
        self.assertEqual(text, "Person at 1.5 meters ahead. Please stop.")
        self.assertEqual(sig, "STOP|person|CENTER|near")

    def test_stop_message_without_distance(self):
        decision = make_decision("STOP", make_object("person", distance=None))
        text, _ = build_message(decision)
        self.assertEqual(text, "Person ahead. Please stop.")

    def test_move_left_message_names_side(self):
        decision = make_decision("MOVE_LEFT", make_object("person", "RIGHT", 2.0))
        text, sig = build_message(decision)
        self.assertEqual(text, "Obstacle on your right at 2 meters. Move left.")
        self.assertEqual(sig, "MOVE_LEFT|person|RIGHT|mid")

    def test_continue_message(self):
        text, sig = build_message(make_decision("CONTINUE"))
        self.assertEqual(text, "Path clear. Continue forward.")
        self.assertEqual(sig, "CONTINUE|||none")

    def test_distance_rounding_half_meter(self):
        self.assertEqual(round_distance(2.1), 2.0)
        self.assertEqual(round_distance(2.3), 2.5)
        self.assertEqual(round_distance(None), None)

    def test_singular_meter_grammar(self):
        decision = make_decision("STOP", make_object("person", "CENTER", 0.9))
        text, _ = build_message(decision)
        self.assertEqual(text, "Person at 1 meter ahead. Please stop.")

    def test_distance_buckets(self):
        self.assertEqual(distance_bucket(0.8), "near")
        self.assertEqual(distance_bucket(2.0), "mid")
        self.assertEqual(distance_bucket(5.0), "far")
        self.assertEqual(distance_bucket(None), "unknown")

    def test_signature_stable_across_jitter(self):
        """0.2 m depth jitter must not change the state signature."""
        a = build_message(make_decision("STOP", make_object("person", distance=1.9)))
        b = build_message(make_decision("STOP", make_object("person", distance=2.1)))
        self.assertEqual(a[1], b[1])
        # Rounding collapses both to the same 2.0 m clause: full anti-jitter.
        self.assertEqual(a[0], b[0])


class TestAnnouncerGating(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.ann = Announcer(clock=self.clock)
        self.stop_dec = make_decision("STOP", make_object("person", "CENTER", 1.4))

    def test_first_event_announces(self):
        self.assertEqual(self.ann.evaluate(self.stop_dec), "Person at 1.5 meters ahead. Please stop.")

    def test_identical_repeat_suppressed_within_cooldown(self):
        self.ann.evaluate(self.stop_dec)
        self.clock.advance(0.5)  # STOP cooldown is 1.0s
        self.assertIsNone(self.ann.evaluate(self.stop_dec))

    def test_identical_repeat_allowed_after_cooldown(self):
        self.ann.evaluate(self.stop_dec)
        self.clock.advance(1.1)
        self.assertEqual(self.ann.evaluate(self.stop_dec), "Person at 1.5 meters ahead. Please stop.")

    def test_different_action_after_short_gap_is_suppressed(self):
        """CONTINUE right after STOP must NOT speak (anti-jitter)."""
        self.ann.evaluate(self.stop_dec)
        self.clock.advance(0.5)
        cont = make_decision("CONTINUE")
        self.assertIsNone(self.ann.evaluate(cont))

    def test_priority_escalation_bypasses_cooldown(self):
        """SLOW_DOWN -> STOP within cooldown is a safety escalation: allowed."""
        self.ann.evaluate(make_decision("SLOW_DOWN", make_object("person", distance=2.5)))
        self.clock.advance(0.5)  # still inside SLOW_DOWN cooldown (3s)
        stop = make_decision("STOP", make_object("person", distance=1.4))
        self.assertEqual(self.ann.evaluate(stop), "Person at 1.5 meters ahead. Please stop.")

    def test_de_escalation_does_not_bypass_cooldown(self):
        """STOP -> MOVE_LEFT shortly after must stay suppressed."""
        self.ann.evaluate(self.stop_dec)
        self.clock.advance(0.5)
        move = make_decision("MOVE_LEFT", make_object("chair", "RIGHT", 1.0))
        self.assertIsNone(self.ann.evaluate(move))

    def test_same_obstacle_much_closer_escalates(self):
        """Person 2.5m announced, then 1.8m: same signature bucket? No - 2.5=mid, 1.8=mid, same sig."""
        # same bucket => same signature; cooldown applies unless distance dropped >= 0.5m
        self.ann.evaluate(make_decision("STOP", make_object("person", "CENTER", 2.5)))
        self.clock.advance(0.4)  # inside 1.0s STOP cooldown
        closer = make_decision("STOP", make_object("person", "CENTER", 1.8))
        self.assertIsNotNone(self.ann.evaluate(closer))  # distance escalation

    def test_small_distance_change_does_not_escalate(self):
        self.ann.evaluate(make_decision("STOP", make_object("person", "CENTER", 2.5)))
        self.clock.advance(0.2)
        slightly = make_decision("STOP", make_object("person", "CENTER", 2.4))
        self.assertIsNone(self.ann.evaluate(slightly))

    def test_reset_clears_state(self):
        self.ann.evaluate(self.stop_dec)
        self.ann.reset()
        self.assertIsNone(self.ann.last_signature)
        self.assertEqual(self.ann.last_announce_time, 0.0)


class TestSpeakerQueue(unittest.TestCase):
    """Queue contract without spinning up the SAPI engine."""

    def test_speak_enqueues_nonblocking(self):
        s = Speaker()
        self.assertTrue(s.speak("Path clear."))
        self.assertEqual(s._queue.qsize(), 1)

    def test_speak_empty_text_rejected(self):
        s = Speaker()
        self.assertFalse(s.speak(""))

    def test_queue_full_drops_message(self):
        from config.audio import SPEECH_QUEUE_SIZE
        s = Speaker()
        for i in range(SPEECH_QUEUE_SIZE):
            self.assertTrue(s.speak(f"msg {i}"))
        self.assertFalse(s.speak("overflow"))  # dropped, no exception

    def test_not_talking_before_start(self):
        s = Speaker()
        self.assertFalse(s.is_talking())


if __name__ == "__main__":
    unittest.main(verbosity=2)
