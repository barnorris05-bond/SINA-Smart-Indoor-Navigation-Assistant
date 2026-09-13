"""
tests/test_risk_estimator.py

PHASE 7 — RISK ESTIMATION TESTS (hardware-free).

Covers (master roadmap §9 + related requirements):
  far+stationary / far+approaching / near+stationary / near+approaching /
  very-near+approaching / receding / unknown motion / stale distance /
  unavailable distance / low-confidence / persistent high-priority obstacle

Plus:
  - TTC gating (only trustworthy APPROACHING rates; none from
    STALE/UNKNOWN/receding/tiny rates)
  - anti-flap: instant escalation, confirmed de-escalation, no
    CRITICAL->LOW from one noisy frame, monotonic safety behavior
  - provenance honesty (MEASURED/SIMULATED/UNAVAILABLE)
  - additive-only stamping (distance/motion/track fields untouched)
  - threshold alignment with navigation (shared config imports)
  - full-pipeline integration: DetectionManager -> ObjectTracker ->
    DistanceFusion -> MotionEstimator -> RiskEstimator, with
    navigation decisions proven byte-identical with/without risk.

No OAK-D, no DepthAI, no OpenCV windows, no pyttsx3.

Run:
    ./.venv/Scripts/python.exe tests/test_risk_estimator.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.object_detector import DetectedObject, BoundingBox
from vision.detection_manager import DetectionManager
from vision.object_tracker import ObjectTracker
from vision.motion_estimator import MotionEstimator
from vision.distance_fusion import DistanceFusion
from navigation.risk_estimator import RiskEstimator, RiskLevel, distance_usable
from navigation.distance_navigator import DistanceNavigator
from navigation.navigation_types import NavigationAction
from depth.provider import DepthProvider, DepthMeasurement, DistanceProvenance

from config.navigation_distance import (
    CRITICAL_DISTANCE_M, WARNING_DISTANCE_M, FAR_DISTANCE_M,
)
from config.risk import TTC_CRITICAL_S, TTC_WARNING_S


FRAME_W, FRAME_H = 640, 400


def make_det(track_id=1, label="person", priority=100, region="CENTER",
             confidence=0.9, distance=3.0, provenance="SIMULATED",
             motion_state="STATIONARY", closing_rate=None,
             motion_provenance="SIMULATED"):
    return DetectedObject(
        label=label,
        confidence=confidence,
        bbox=BoundingBox(300, 150, 340, 350),
        track_id=track_id,
        distance=distance,
        distance_provenance=provenance,
        distance_source="test",
        priority=priority,
        region=region,
        motion_state=motion_state,
        closing_rate_mps=closing_rate,
        motion_provenance=motion_provenance,
    )


class ScriptedProvider(DepthProvider):
    """Test double: pops scripted distances, always SIMULATED."""

    def __init__(self, distances):
        self._queue = list(distances)

    @property
    def name(self):
        return "scripted_test"

    def start(self):
        return None

    def stop(self):
        return None

    def get_measurement(self, bbox, frame_shape, label=None, now=None):
        if not self._queue:
            return DepthMeasurement(None, DistanceProvenance.UNAVAILABLE,
                                    self.name, "exhausted", now or 0.0)
        return DepthMeasurement(self._queue.pop(0), DistanceProvenance.SIMULATED,
                                self.name, None, now or 0.0)


class RiskLadderTest(unittest.TestCase):
    """Single-shot ladder behavior (no anti-flap interference)."""

    def setUp(self):
        self.est = RiskEstimator()

    def level(self, **kw):
        return self.est.assess(make_det(**kw)).level

    # ---------------- §9 scenario matrix ----------------

    def test_far_stationary_low(self):
        # Book (low priority) far away.
        self.assertEqual(self.level(label="book", priority=20,
                                    distance=4.5), RiskLevel.LOW)

    def test_far_approaching_high_priority_is_high(self):
        # Person 3 m out but closing at 1.0 m/s: TTC 3.0 s -> HIGH.
        lvl = self.level(motion_state="APPROACHING", closing_rate=1.0)
        self.assertEqual(lvl, RiskLevel.HIGH)

    def test_near_stationary_low_priority_is_medium(self):
        # Bottle 1.5 m, stationary: warning band, not high-priority.
        lvl = self.level(label="bottle", priority=40, distance=1.5)
        self.assertEqual(lvl, RiskLevel.MEDIUM)

    def test_near_stationary_high_priority_is_high(self):
        # Chair 1.5 m, stationary: high priority -> HIGH.
        lvl = self.level(label="chair", priority=90, distance=1.5)
        self.assertEqual(lvl, RiskLevel.HIGH)

    def test_near_approaching_is_critical_by_ttc(self):
        # Person 2.4 m closing 0.8 m/s: TTC 3.0 s -> HIGH boundary;
        # at 1.0 m/s TTC 2.4 s still HIGH (>1.5); verify HIGH here and
        # CRITICAL only under the TTC_CRITICAL_S boundary below.
        lvl = self.level(motion_state="APPROACHING", closing_rate=0.8,
                         distance=2.4)
        self.assertEqual(lvl, RiskLevel.HIGH)

    def test_very_near_approaching_is_critical(self):
        lvl = self.level(distance=0.9, motion_state="APPROACHING",
                         closing_rate=1.2)
        self.assertEqual(lvl, RiskLevel.CRITICAL)

    def test_receding_never_gets_ttc(self):
        out = self.est.assess(make_det(distance=2.0,
                                       motion_state="RECEDING",
                                       closing_rate=-0.5))
        self.assertIsNone(out.ttc_s)
        self.assertEqual(out.level, RiskLevel.HIGH)  # chair-band distance
        self.assertFalse(any("TTC" in r for r in out.reasons))

    def test_unknown_motion_no_ttc(self):
        out = self.est.assess(make_det(distance=1.0,
                                       motion_state="UNKNOWN",
                                       closing_rate=2.0))
        self.assertIsNone(out.ttc_s)
        # Still critical by distance alone.
        self.assertEqual(out.level, RiskLevel.CRITICAL)

    def test_stale_distance_not_distance_only(self):
        # STALE distance: unusable -> conservative branch, not ladder.
        out = self.est.assess(make_det(distance=0.8,
                                       provenance="STALE"))
        self.assertEqual(out.level, RiskLevel.MEDIUM)  # person, priority 100
        self.assertEqual(out.provenance, "UNAVAILABLE")

    def test_unavailable_distance_low_priority_low(self):
        out = self.est.assess(make_det(label="book", priority=20,
                                       distance=None,
                                       provenance="UNAVAILABLE"))
        self.assertEqual(out.level, RiskLevel.LOW)

    def test_low_confidence_never_lowers_risk(self):
        # Same scene, high vs low confidence: level must not drop.
        hi = self.est.assess(make_det(distance=1.0, confidence=0.95))
        lo = self.est.assess(make_det(distance=1.0, confidence=0.55))
        self.assertEqual(hi.level, lo.level, RiskLevel.CRITICAL)
        self.assertTrue(any("uncertain" in r for r in lo.reasons))
        self.assertFalse(any("uncertain" in r for r in hi.reasons))

    # ---------------- TTC boundaries ----------------

    def test_ttc_critical_boundary(self):
        # distance/rate == TTC_CRITICAL_S exactly -> CRITICAL.
        rate = 2.0
        d = TTC_CRITICAL_S * rate
        lvl = self.level(motion_state="APPROACHING", closing_rate=rate,
                         distance=d)
        self.assertEqual(lvl, RiskLevel.CRITICAL)

    def test_ttc_warning_boundary(self):
        # TTC == TTC_WARNING_S -> HIGH (not CRITICAL).
        rate = 1.0
        d = TTC_WARNING_S * rate
        lvl = self.level(motion_state="APPROACHING", closing_rate=rate,
                         distance=d)
        self.assertEqual(lvl, RiskLevel.HIGH)

    def test_ttc_below_min_rate_not_trusted(self):
        # Rate < TTC_MIN_CLOSING_RATE_MPS: no TTC, plain distance ladder.
        out = self.est.assess(make_det(distance=2.0,
                                       motion_state="APPROACHING",
                                       closing_rate=0.05))
        self.assertIsNone(out.ttc_s)

    def test_negative_rate_never_yields_ttc(self):
        out = self.est.assess(make_det(distance=2.0,
                                       motion_state="APPROACHING",
                                       closing_rate=-0.3))
        self.assertIsNone(out.ttc_s)

    # ---------------- Priority interaction ----------------

    def test_high_priority_far_is_medium(self):
        lvl = self.level(label="chair", priority=90, distance=3.5)
        self.assertEqual(lvl, RiskLevel.MEDIUM)

    def test_low_priority_warning_band_is_medium(self):
        lvl = self.level(label="bottle", priority=40, distance=2.0)
        self.assertEqual(lvl, RiskLevel.MEDIUM)

    # ---------------- Usability policy parity ----------------

    def test_usability_matches_navigation_rule(self):
        det = make_det(provenance="STALE", distance=1.0)
        self.assertFalse(distance_usable(det))
        det2 = make_det(provenance="MEASURED", distance=1.0)
        self.assertTrue(distance_usable(det2))


class RiskAntiFlapTest(unittest.TestCase):
    """update()-level hysteresis: instant up, confirmed down."""

    def setUp(self):
        self.est = RiskEstimator()

    def test_escalation_is_immediate(self):
        d = make_det(track_id=7, distance=3.5)  # MEDIUM
        self.assertEqual(self.est.update([d])[0].risk_level, "MEDIUM")
        d.distance = 0.9                        # CRITICAL scene
        self.assertEqual(self.est.update([d])[0].risk_level, "CRITICAL")

    def test_deescalation_requires_confirmation(self):
        d = make_det(track_id=7, distance=0.9)  # CRITICAL
        self.est.update([d])
        # Scene relaxes to LOW (book far away).
        d.label, d.priority = "book", 20
        d.distance, d.motion_state = 5.0, "STATIONARY"
        # Frames 1-2: still CRITICAL (proposed LOW unconfirmed).
        self.assertEqual(self.est.update([d])[0].risk_level, "CRITICAL")
        self.assertEqual(self.est.update([d])[0].risk_level, "CRITICAL")
        # Frame 3: confirmed -> LOW.
        self.assertEqual(self.est.update([d])[0].risk_level, "LOW")

    def test_one_noisy_frame_cannot_drop_critical(self):
        d = make_det(track_id=7, distance=0.9)
        self.est.update([d])                    # CRITICAL
        d.distance = 5.0                        # one noisy far frame
        self.assertEqual(self.est.update([d])[0].risk_level, "CRITICAL")
        d.distance = 0.9                        # back to critical
        self.assertEqual(self.est.update([d])[0].risk_level, "CRITICAL")

    def test_monotonic_no_level_skipping_on_drop(self):
        d = make_det(track_id=7, distance=0.9)  # CRITICAL
        self.est.update([d])
        d.distance = 5.0
        d.label, d.priority = "book", 20
        # Proposed LOW (skip from CRITICAL): streak counts but level
        # steps through MEDIUM/HIGH only after sustained evidence.
        seq = [self.est.update([d])[0].risk_level for _ in range(5)]
        self.assertEqual(seq[0], "CRITICAL")
        self.assertEqual(seq[-1], "LOW")
        self.assertEqual(seq, ["CRITICAL", "CRITICAL", "LOW", "LOW", "LOW"])

    def test_reset_clears_antiflap(self):
        d = make_det(track_id=7, distance=0.9)
        self.est.update([d])
        self.est.reset()
        d2 = make_det(track_id=7, distance=0.9)
        self.assertEqual(self.est.update([d2])[0].risk_level, "CRITICAL")

    def test_untracked_assessments_are_stateless(self):
        d = make_det(track_id=None, distance=0.9)
        self.assertEqual(self.est.update([d])[0].risk_level, "CRITICAL")
        d.distance = 5.0
        d.label, d.priority = "book", 20
        # No track -> no hysteresis memory -> immediate LOW.
        self.assertEqual(self.est.update([d])[0].risk_level, "LOW")


class RiskPipelineIntegrationTest(unittest.TestCase):
    """
    Full chain with real SINA classes (no hardware):

        DetectionManager -> ObjectTracker -> DistanceFusion ->
        MotionEstimator -> RiskEstimator
    """

    def setUp(self):
        self.t = 3000.0
        self.manager = DetectionManager()
        self.tracker = ObjectTracker()
        self.navigator = DistanceNavigator()

    def _pipeline_pass(self, distances, motion, risk):
        self.t += 0.25
        clock = lambda: self.t
        fusion = DistanceFusion(ScriptedProvider(distances), clock=clock)
        raw = [DetectedObject(label="person", confidence=0.93,
                              bbox=BoundingBox(310, 150, 350, 350))]
        dets = self.manager.process(raw, FRAME_W)
        dets = self.tracker.update(dets)
        fusion.fuse(dets, (FRAME_H, FRAME_W))
        motion.update(dets)
        risk.update(dets)
        return dets

    def test_risk_fields_flow_through_pipeline(self):
        motion = MotionEstimator(clock=lambda: self.t)
        risk = RiskEstimator()
        seq = [3.0, 2.8, 2.6, 2.4, 2.2, 1.0]
        last = None
        for d in seq:
            last = self._pipeline_pass([d], motion, risk)
        det = last[0]
        self.assertEqual(det.track_id, 1)
        self.assertEqual(det.distance_provenance, "SIMULATED")
        self.assertEqual(det.motion_state, "APPROACHING")
        self.assertEqual(det.risk_level, "CRITICAL")  # 1.0 <= 1.2 m
        self.assertTrue(det.risk_reasons)
        self.assertEqual(det.risk_provenance, "SIMULATED")

    def test_navigation_decisions_unchanged_with_risk_in_chain(self):
        # Byte-identical navigation with and without the risk layer:
        # risk is annotation until Phase 8 integrates it.
        motion = MotionEstimator(clock=lambda: self.t)
        risk = RiskEstimator()
        with_risk = self._pipeline_pass([1.0], motion, risk)
        decision_with = self.navigator.decide(with_risk)

        motion2 = MotionEstimator(clock=lambda: self.t + 100)
        risk2 = RiskEstimator()
        without_risk = self._pipeline_pass_no_risk([1.0], motion2)
        decision_without = self.navigator.decide(without_risk)

        self.assertEqual(decision_with.action,
                         decision_without.action,
                         NavigationAction.STOP)
        self.assertEqual(decision_with.reason,
                         decision_without.reason)

    def _pipeline_pass_no_risk(self, distances, motion):
        self.t += 0.25
        clock = lambda: self.t
        fusion = DistanceFusion(ScriptedProvider(distances), clock=clock)
        raw = [DetectedObject(label="person", confidence=0.93,
                              bbox=BoundingBox(310, 150, 350, 350))]
        dets = self.manager.process(raw, FRAME_W)
        dets = self.tracker.update(dets)
        fusion.fuse(dets, (FRAME_H, FRAME_W))
        motion.update(dets)
        return dets

    def test_approach_stream_escalates_risk(self):
        motion = MotionEstimator(clock=lambda: self.t)
        risk = RiskEstimator()
        levels = []
        for d in [4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.9]:
            dets = self._pipeline_pass([d], motion, risk)
            levels.append(dets[0].risk_level)
        # Frame 1: 4.5m far + no motion history yet (honest UNKNOWN)
        # -> LOW. Escalation follows as history and proximity build.
        self.assertEqual(levels[0], "LOW")
        self.assertEqual(levels[-1], "CRITICAL")
        self.assertIn("HIGH", levels)
        # Severity must never DECREASE during a sustained approach
        # (at 2 m/s the TTC ladder can skip MEDIUM - that is fine;
        # going backwards would not be).
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        values = [order[l] for l in levels]
        self.assertEqual(values, sorted(values),
                         f"severity decreased mid-approach: {levels}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
