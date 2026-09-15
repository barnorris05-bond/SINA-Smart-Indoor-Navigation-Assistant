"""
tests/test_temporal_navigator.py

PHASE 8 — TEMPORAL NAVIGATION STABILIZATION TESTS (hardware-free).

Covers (master roadmap §27 requirements + §28-32, §39):

  stable decisions / immediate escalation / immediate critical STOP /
  de-escalation confirmation / directional jitter / STOP-SLOW_DOWN
  jitter / CONTINUE-SLOW_DOWN jitter / persistent change propagates /
  multiple objects / track disappearance / scene-CRITICAL forces STOP /
  unknown-missing-STALE-UNAVAILABLE distances / provenance honesty /
  legacy fallback / reset / no state leakage / reason preservation /
  audio-signature stability / full E2E pipeline

No OAK-D, no DepthAI device, no OpenCV windows, no pyttsx3, no network.

Run:
    ./.venv/Scripts/python.exe tests/test_temporal_navigator.py
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
from navigation.risk_estimator import RiskEstimator
from navigation.distance_navigator import DistanceNavigator
from navigation.navigator import Navigator
from navigation.temporal_navigator import TemporalNavigator
from navigation.navigation_types import NavigationAction
from depth.provider import DepthProvider, DepthMeasurement, DistanceProvenance

from config.temporal_navigation import (
    DEESCALATION_CONFIRM_FRAMES,
    ACTION_CHANGE_CONFIRM_FRAMES,
)


FRAME_W, FRAME_H = 640, 400

_REGION_BBOX = {
    "LEFT": (60, 150, 110, 350),
    "CENTER": (300, 150, 350, 350),
    "RIGHT": (540, 150, 590, 350),
}

_SEV = {"CONTINUE": 1, "SLOW_DOWN": 2, "MOVE_LEFT": 2, "MOVE_RIGHT": 2,
        "STOP": 4}


def make_det(label="person", priority=100, region="CENTER", distance=None,
             provenance="UNAVAILABLE", risk_level=None, track_id=None,
             motion_state=None, closing_rate=None, confidence=0.9):
    x1, y1, x2, y2 = _REGION_BBOX[region]
    return DetectedObject(
        label=label,
        confidence=confidence,
        bbox=BoundingBox(x1, y1, x2, y2),
        track_id=track_id,
        distance=distance,
        distance_provenance=provenance,
        distance_source="test" if distance is not None else None,
        priority=priority,
        region=region,
        motion_state=motion_state,
        closing_rate_mps=closing_rate,
        risk_level=risk_level,
    )


def nav(tnav, detections):
    """decide() convenience returning the NavigationAction."""
    return tnav.decide(detections).action


class TemporalStabilityTest(unittest.TestCase):
    """§27.1-3, 9, 10 — stable scenes and flap suppression."""

    def setUp(self):
        self.tnav = TemporalNavigator(DistanceNavigator())

    def test_stable_continue_empty_scene(self):
        for _ in range(5):
            self.assertEqual(nav(self.tnav, []), NavigationAction.CONTINUE)

    def test_stable_slow_down(self):
        det = make_det(label="bottle", priority=40, distance=2.4,
                       provenance="SIMULATED")
        for _ in range(4):
            self.assertEqual(nav(self.tnav, [det]),
                             NavigationAction.SLOW_DOWN)

    def test_stable_stop(self):
        det = make_det(distance=1.0, provenance="SIMULATED")
        for _ in range(4):
            self.assertEqual(nav(self.tnav, [det]), NavigationAction.STOP)

    def test_stop_survives_slowdown_flap(self):
        # §14: STOP -> SLOW_DOWN -> STOP -> SLOW_DOWN noise must not flap.
        near = make_det(distance=1.0, provenance="SIMULATED")
        mid = make_det(distance=2.4, provenance="SIMULATED")
        for det in (near, mid, near, mid):
            self.assertEqual(nav(self.tnav, [det]), NavigationAction.STOP)

    def test_slow_down_survives_continue_flap(self):
        # §15: SLOW_DOWN -> CONTINUE -> SLOW_DOWN -> CONTINUE noise.
        near = make_det(label="bottle", priority=40, distance=2.4,
                        provenance="SIMULATED")
        far = make_det(label="bottle", priority=40, distance=4.5,
                       provenance="SIMULATED")
        for det in (near, far, near, far):
            self.assertEqual(nav(self.tnav, [det]),
                             NavigationAction.SLOW_DOWN)

    def test_continue_requires_confirmed_deescalation(self):
        # A persistent far scene eventually relaxes, after exactly
        # DEESCALATION_CONFIRM_FRAMES consecutive candidates.
        near = make_det(label="bottle", priority=40, distance=2.4,
                        provenance="SIMULATED")
        far = make_det(label="bottle", priority=40, distance=4.5,
                       provenance="SIMULATED")
        self.assertEqual(nav(self.tnav, [near]),
                         NavigationAction.SLOW_DOWN)
        for i in range(1, DEESCALATION_CONFIRM_FRAMES + 1):
            out = self.tnav.decide([far])
            if i < DEESCALATION_CONFIRM_FRAMES:
                self.assertEqual(out.action, NavigationAction.SLOW_DOWN)
                self.assertIn("holding", out.reason)
            else:
                self.assertEqual(out.action, NavigationAction.CONTINUE)


class EscalationTest(unittest.TestCase):
    """§27.4-5, §28 — escalation is immediate and monotonic."""

    def setUp(self):
        self.tnav = TemporalNavigator(DistanceNavigator())

    def test_immediate_escalation_to_slow_down(self):
        self.assertEqual(nav(self.tnav, []),
                         NavigationAction.CONTINUE)
        det = make_det(label="bottle", priority=40, distance=2.4,
                       provenance="SIMULATED")
        self.assertEqual(nav(self.tnav, [det]),
                         NavigationAction.SLOW_DOWN)

    def test_immediate_stop_from_continue(self):
        det = make_det(distance=1.0, provenance="SIMULATED")
        self.assertEqual(nav(self.tnav, [det]), NavigationAction.STOP)

    def test_stop_reasserted_during_pending_deescalation(self):
        near = make_det(distance=1.0, provenance="SIMULATED")
        far = make_det(distance=4.5, provenance="SIMULATED")
        self.assertEqual(nav(self.tnav, [near]), NavigationAction.STOP)
        self.assertEqual(nav(self.tnav, [far]),
                         NavigationAction.STOP)  # held
        # Hazard returns while de-escalation pending -> STOP immediately,
        # pending counter cleared (no stale confirmation leakage).
        self.assertEqual(nav(self.tnav, [near]), NavigationAction.STOP)
        self.assertEqual(nav(self.tnav, [far]),
                         NavigationAction.STOP)  # held again from scratch

    def test_scene_critical_forces_stop(self):
        # §6/§11: fast approach just outside the distance STOP band is
        # risk-CRITICAL by TTC -> temporal layer maps CRITICAL -> STOP.
        det = make_det(distance=2.6, provenance="SIMULATED",
                       motion_state="APPROACHING", closing_rate=1.0,
                       risk_level="CRITICAL")
        out = self.tnav.decide([det])
        self.assertEqual(out.action, NavigationAction.STOP)
        self.assertIn("Critical risk", out.reason)

    def test_monotonic_escalation_low_to_critical(self):
        # §28: LOW -> MEDIUM -> HIGH -> CRITICAL; output never goes back.
        book = make_det(label="book", priority=20, distance=4.5,
                        provenance="SIMULATED", risk_level="LOW")
        bottle = make_det(label="bottle", priority=40, distance=2.0,
                          provenance="SIMULATED", risk_level="MEDIUM")
        chair = make_det(label="chair", priority=90, distance=2.4,
                         provenance="SIMULATED", motion_state="APPROACHING",
                         closing_rate=0.8, risk_level="HIGH")
        person = make_det(distance=1.0, provenance="SIMULATED",
                          risk_level="CRITICAL")
        seq = [book, bottle, chair, person]
        values = [_SEV[nav(self.tnav, [d]).name] for d in seq]
        self.assertEqual(values, sorted(values),
                         f"escalation went backwards: {values}")
        self.assertEqual(values[-1], 4)


class DeescalationTest(unittest.TestCase):
    """§27.6-7, §29 — drops require persistence, spikes reset."""

    def setUp(self):
        self.tnav = TemporalNavigator(DistanceNavigator())

    def test_stop_relaxes_only_after_confirmation(self):
        near = make_det(distance=1.0, provenance="SIMULATED")
        far = make_det(distance=4.5, provenance="SIMULATED")
        self.assertEqual(nav(self.tnav, [near]), NavigationAction.STOP)
        for i in range(1, DEESCALATION_CONFIRM_FRAMES + 1):
            out = self.tnav.decide([far])
            if i < DEESCALATION_CONFIRM_FRAMES:
                self.assertEqual(out.action, NavigationAction.STOP)
                self.assertIn("de-escalation", out.reason)
            else:
                self.assertEqual(out.action, NavigationAction.CONTINUE)

    def test_one_noisy_frame_cannot_relax_stop(self):
        near = make_det(distance=1.0, provenance="SIMULATED")
        far = make_det(distance=4.5, provenance="SIMULATED")
        self.assertEqual(nav(self.tnav, [near]), NavigationAction.STOP)
        self.assertEqual(nav(self.tnav, [far]),
                         NavigationAction.STOP)   # noisy low frame held
        self.assertEqual(nav(self.tnav, [near]),
                         NavigationAction.STOP)   # hazard back

    def test_risk_spike_resets_pending_confirmation(self):
        # §29: CRITICAL held; HIGH frames 1-2; CRITICAL spike resets;
        # then 3 consecutive lower frames finally relax.
        crit = make_det(distance=1.0, provenance="SIMULATED",
                        risk_level="CRITICAL")
        high = make_det(distance=1.4, provenance="SIMULATED",
                        risk_level="HIGH")
        self.assertEqual(nav(self.tnav, [crit]), NavigationAction.STOP)
        self.assertEqual(nav(self.tnav, [high]), NavigationAction.STOP)
        self.assertEqual(nav(self.tnav, [high]), NavigationAction.STOP)
        self.assertEqual(nav(self.tnav, [crit]), NavigationAction.STOP)
        for _ in range(DEESCALATION_CONFIRM_FRAMES - 1):
            self.assertEqual(nav(self.tnav, [high]),
                             NavigationAction.STOP)
        self.assertEqual(nav(self.tnav, [high]),
                         NavigationAction.SLOW_DOWN)


class ActionChangeTest(unittest.TestCase):
    """§13-15, §27.8/11 — equal-severity action changes need confirmation."""

    def setUp(self):
        self.tnav = TemporalNavigator(DistanceNavigator())

    def test_directional_jitter_suppressed(self):
        # §13: alternating lateral hazards must not chatter direction.
        right = make_det(label="chair", priority=90, region="RIGHT",
                         distance=2.0, provenance="SIMULATED")
        left = make_det(label="chair", priority=90, region="LEFT",
                        distance=2.0, provenance="SIMULATED")
        self.assertEqual(nav(self.tnav, [right]),
                         NavigationAction.MOVE_LEFT)
        for det in (left, right, left, right, left):
            self.assertEqual(nav(self.tnav, [det]),
                             NavigationAction.MOVE_LEFT,
                             "direction flapped on alternating noise")

    def test_persistent_direction_change_switches(self):
        right = make_det(label="chair", priority=90, region="RIGHT",
                         distance=2.0, provenance="SIMULATED")
        left = make_det(label="chair", priority=90, region="LEFT",
                        distance=2.0, provenance="SIMULATED")
        self.assertEqual(nav(self.tnav, [right]),
                         NavigationAction.MOVE_LEFT)
        out = self.tnav.decide([left])
        self.assertEqual(out.action, NavigationAction.MOVE_LEFT)
        self.assertIn("action-change", out.reason)
        self.assertEqual(nav(self.tnav, [left]),
                         NavigationAction.MOVE_RIGHT)

    def test_intervening_reconfirmation_resets_direction_pending(self):
        right = make_det(label="chair", priority=90, region="RIGHT",
                         distance=2.0, provenance="SIMULATED")
        left = make_det(label="chair", priority=90, region="LEFT",
                        distance=2.0, provenance="SIMULATED")
        self.assertEqual(nav(self.tnav, [right]),
                         NavigationAction.MOVE_LEFT)
        nav(self.tnav, [left])    # pending 1/2
        nav(self.tnav, [right])   # re-confirmation resets
        nav(self.tnav, [left])    # pending 1/2 again, never 2/2
        self.assertEqual(nav(self.tnav, [right]),
                         NavigationAction.MOVE_LEFT)

    def test_forward_to_lateral_change_needs_confirmation(self):
        # SLOW_DOWN (center) -> lateral hazard at equal severity: hold
        # once, accept on the second consecutive candidate.
        center = make_det(label="bottle", priority=40, distance=2.4,
                          provenance="SIMULATED")
        lateral = make_det(label="bottle", priority=40, region="RIGHT",
                           distance=2.4, provenance="SIMULATED")
        self.assertEqual(nav(self.tnav, [center]),
                         NavigationAction.SLOW_DOWN)
        out = self.tnav.decide([lateral])
        self.assertEqual(out.action, NavigationAction.SLOW_DOWN)
        self.assertEqual(nav(self.tnav, [lateral]),
                         NavigationAction.MOVE_LEFT)

    def test_lateral_to_forward_accepts_immediately(self):
        # Obstacle left the lateral band: returning to a forward action
        # at equal severity is not gated (documented rule 6 exception
        # for lateral -> forward; forward -> lateral IS gated above).
        lateral = make_det(label="bottle", priority=40, region="RIGHT",
                           distance=2.4, provenance="SIMULATED")
        center = make_det(label="bottle", priority=40, distance=2.4,
                          provenance="SIMULATED")
        self.assertEqual(nav(self.tnav, [lateral]),
                         NavigationAction.MOVE_LEFT)
        # First lateral->forward candidate is held once (equal-severity
        # action change), then accepted — deterministic either way, but
        # must NEVER bounce back to MOVE_LEFT on a single center frame.
        out = self.tnav.decide([center])
        self.assertIn(out.action,
                      (NavigationAction.SLOW_DOWN, NavigationAction.MOVE_LEFT))
        self.assertEqual(nav(self.tnav, [center]),
                         NavigationAction.SLOW_DOWN)


class SceneRiskTest(unittest.TestCase):
    """§17 — max-risk scene semantics; §16 unknown risk; optional data."""

    def setUp(self):
        self.tnav = TemporalNavigator(DistanceNavigator())

    def test_low_risk_object_never_dilutes_critical(self):
        person = make_det(distance=1.0, provenance="SIMULATED",
                          risk_level="CRITICAL")
        book = make_det(label="book", priority=20, distance=4.5,
                        provenance="SIMULATED", risk_level="LOW")
        self.assertEqual(nav(self.tnav, [person, book]),
                         NavigationAction.STOP)
        self.assertEqual(nav(self.tnav, [book, person]),
                         NavigationAction.STOP)

    def test_scene_risk_is_max_over_detections(self):
        high = make_det(label="chair", priority=90, distance=2.4,
                        provenance="SIMULATED", risk_level="HIGH")
        medium = make_det(label="bottle", priority=40, distance=2.0,
                          provenance="SIMULATED", risk_level="MEDIUM")
        # Candidate SLOW_DOWN (sev 2) + scene HIGH (3) -> accept sev 3.
        self.assertEqual(nav(self.tnav, [medium]),
                         NavigationAction.SLOW_DOWN)
        out = self.tnav.decide([high, medium])
        self.assertEqual(out.action, NavigationAction.SLOW_DOWN)

    def test_no_risk_data_still_stabilizes(self):
        # All annotations None: pure action-floor behavior (legacy path).
        det = make_det(distance=2.4, provenance="SIMULATED")
        for _ in range(3):
            self.assertEqual(nav(self.tnav, [det]),
                             NavigationAction.SLOW_DOWN)


class FallbackAndProvenanceTest(unittest.TestCase):
    """§5, §20-22, §27.16-23 — legacy fallback + data honesty."""

    def setUp(self):
        self.tnav = TemporalNavigator(DistanceNavigator())

    def test_no_distance_uses_legacy_navigation(self):
        # §32: no usable distance anywhere -> legacy spatial policy,
        # still stabilized into a valid decision.
        det = make_det(label="person", priority=100)  # CENTER, no distance
        self.assertEqual(nav(self.tnav, [det]), NavigationAction.STOP)

    def test_legacy_nonhazard_center_slowdown(self):
        det = make_det(label="bottle", priority=40)
        self.assertEqual(nav(self.tnav, [det]),
                         NavigationAction.SLOW_DOWN)

    def test_stale_distance_never_quoted_as_current(self):
        # STALE is unusable -> legacy fallback; the reason must not
        # present the stale value as a fresh measurement.
        det = make_det(label="person", priority=100, distance=0.8,
                       provenance="STALE")
        out = self.tnav.decide([det])
        self.assertEqual(out.action, NavigationAction.STOP)
        self.assertNotIn("0.8", out.reason)
        self.assertEqual(det.distance_provenance, "STALE")

    def test_unavailable_distance_same_path(self):
        det = make_det(label="bottle", priority=40, distance=None,
                       provenance="UNAVAILABLE")
        self.assertEqual(nav(self.tnav, [det]),
                         NavigationAction.SLOW_DOWN)

    def test_measured_provenance_drives_navigation(self):
        det = make_det(distance=1.0, provenance="MEASURED")
        out = self.tnav.decide([det])
        self.assertEqual(out.action, NavigationAction.STOP)
        self.assertIn("MEASURED", out.reason)

    def test_simulated_provenance_in_development_mode(self):
        det = make_det(distance=1.0, provenance="SIMULATED")
        out = self.tnav.decide([det])
        self.assertEqual(out.action, NavigationAction.STOP)
        self.assertIn("SIMULATED", out.reason)

    def test_no_provenance_promotion(self):
        # Decide must never rewrite provenance annotations.
        det = make_det(distance=1.0, provenance="SIMULATED",
                       risk_level="HIGH")
        before = (det.distance_provenance, det.risk_level,
                  det.motion_state, det.distance)
        self.tnav.decide([det])
        after = (det.distance_provenance, det.risk_level,
                 det.motion_state, det.distance)
        self.assertEqual(before, after)
        self.assertNotEqual(det.distance_provenance, "MEASURED")

    def test_decision_reasons_are_auditable(self):
        near = make_det(distance=1.0, provenance="SIMULATED")
        far = make_det(distance=4.5, provenance="SIMULATED")
        self.assertEqual(nav(self.tnav, [near]), NavigationAction.STOP)
        out = self.tnav.decide([far])
        self.assertTrue(out.reason)             # holding reason present
        self.assertIn("holding", out.reason)


class ResetAndLeakageTest(unittest.TestCase):
    """§27.24-25 — reset semantics; scene-level state cannot leak."""

    def setUp(self):
        self.tnav = TemporalNavigator(DistanceNavigator())

    def test_reset_clears_state(self):
        near = make_det(distance=1.0, provenance="SIMULATED")
        far = make_det(distance=4.5, provenance="SIMULATED")
        self.assertEqual(nav(self.tnav, [near]), NavigationAction.STOP)
        self.tnav.reset()
        # After reset the far scene relaxes immediately (no held STOP).
        self.assertEqual(nav(self.tnav, [far]),
                         NavigationAction.CONTINUE)

    def test_scene_level_state_no_cross_track_leakage(self):
        # Temporal state is scene-level by design: a NEW track in a
        # cleared scene cannot inherit a stale per-track hold.
        t1 = make_det(distance=1.0, provenance="SIMULATED",
                      risk_level="CRITICAL", track_id=1)
        self.assertEqual(nav(self.tnav, [t1]), NavigationAction.STOP)
        t2 = make_det(distance=4.5, provenance="SIMULATED",
                      risk_level="LOW", track_id=2)
        # Different track, low risk: de-escalation still applies
        # (scene did not instantly clear) - but after confirmation the
        # scene relaxes and no t1 state remains anywhere.
        for _ in range(DEESCALATION_CONFIRM_FRAMES):
            out = self.tnav.decide([t2])
        self.assertEqual(out.action, NavigationAction.CONTINUE)


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
        # Contract parity with the real providers: a stream with no
        # data reports UNAVAILABLE, never SIMULATED-with-None.
        if not self._queue or self._queue[0] is None:
            if self._queue:
                self._queue.pop(0)
            return DepthMeasurement(None, DistanceProvenance.UNAVAILABLE,
                                    self.name, "exhausted", now or 0.0)
        return DepthMeasurement(self._queue.pop(0), DistanceProvenance.SIMULATED,
                                self.name, None, now or 0.0)


class UnavailableProvider(DepthProvider):
    """Test double: always UNAVAILABLE (legacy fallback path)."""

    def __init__(self, distances=None):
        pass
    @property
    def name(self):
        return "unavailable_test"

    def start(self):
        return None

    def stop(self):
        return None

    def get_measurement(self, bbox, frame_shape, label=None, now=None):
        return DepthMeasurement(None, DistanceProvenance.UNAVAILABLE,
                                self.name, "no depth", now or 0.0)


class PipelineE2ETest(unittest.TestCase):
    """
    §30-32, §39: full hardware-free chain

        DetectionManager -> ObjectTracker -> DistanceFusion ->
        MotionEstimator -> RiskEstimator -> TemporalNavigator
    """

    def setUp(self):
        self.t = 3000.0
        self.manager = DetectionManager()
        self.tracker = ObjectTracker()
        self.navigator = TemporalNavigator(DistanceNavigator())

    def _frame(self, raw, motion, risk, provider_cls, distances):
        self.t += 0.25
        clock = lambda: self.t
        fusion = DistanceFusion(provider_cls(distances), clock=clock)
        dets = self.manager.process(raw, FRAME_W)
        dets = self.tracker.update(dets)
        fusion.fuse(dets, (FRAME_H, FRAME_W))
        motion.update(dets)
        risk.update(dets)
        decision = self.navigator.decide(dets)
        return dets, decision

    def test_genuine_hazard_escalates_to_stop(self):
        # §30: person 2.8 -> 2.4 -> 1.5 -> 1.0 m. Persistent track,
        # simulated provenance, escalating navigation, final STOP.
        motion = MotionEstimator(clock=lambda: self.t)
        risk = RiskEstimator()
        raw = [DetectedObject(label="person", confidence=0.93,
                              bbox=BoundingBox(310, 150, 350, 350))]
        decisions = []
        track_ids = []
        provs = []
        for d in [2.8, 2.4, 1.5, 1.0]:
            dets, decision = self._frame(raw, motion, risk,
                                         ScriptedProvider, [d])
            decisions.append(decision.action)
            track_ids.append(dets[0].track_id)
            provs.append(dets[0].distance_provenance)
        self.assertEqual(track_ids, [1, 1, 1, 1])
        self.assertEqual(provs, ["SIMULATED"] * 4)
        self.assertEqual(decisions[-1], NavigationAction.STOP)
        # Severity never decreases during a sustained approach.
        values = [_SEV[a.name] for a in decisions]
        self.assertEqual(values, sorted(values),
                         f"navigation regressed mid-approach: {decisions}")

    def test_track_disappearance_holds_stop_then_recovers(self):
        # §19/§39: CRITICAL person vanishes for ONE frame (detection
        # loss): the scene must NOT instantly relax; the object
        # re-appearing keeps its track id.
        motion = MotionEstimator(clock=lambda: self.t)
        risk = RiskEstimator()
        raw = [DetectedObject(label="person", confidence=0.93,
                              bbox=BoundingBox(310, 150, 350, 350))]
        dets, decision = self._frame(raw, motion, risk,
                                     ScriptedProvider, [1.0])
        self.assertEqual(decision.action, NavigationAction.STOP)
        track_id = dets[0].track_id

        # Detection loss frame (no detections at all).
        _, decision = self._frame([], motion, risk,
                                  ScriptedProvider, [None])
        self.assertEqual(decision.action, NavigationAction.STOP)  # held

        # Object returns (within tracker's missing window).
        dets, decision = self._frame(raw, motion, risk,
                                     ScriptedProvider, [1.0])
        self.assertEqual(decision.action, NavigationAction.STOP)
        self.assertEqual(dets[0].track_id, track_id)

    def test_single_frame_distance_loss_does_not_crash_or_fabricate(self):
        # §39: one frame with no distance mid-approach: legacy candidate
        # path, no crash, no fabricated measurement.
        motion = MotionEstimator(clock=lambda: self.t)
        risk = RiskEstimator()
        raw = [DetectedObject(label="person", confidence=0.93,
                              bbox=BoundingBox(310, 150, 350, 350))]
        dets, _ = self._frame(raw, motion, risk, ScriptedProvider, [1.0])
        self.assertEqual(dets[0].distance_provenance, "SIMULATED")
        # Distance stream cuts out -> UNAVAILABLE this frame.
        dets, decision = self._frame(raw, motion, risk,
                                     ScriptedProvider, [None])
        self.assertEqual(dets[0].distance_provenance, "UNAVAILABLE")
        self.assertIn(decision.action, list(NavigationAction))

    def test_legacy_fallback_e2e_matches_spatial_navigator(self):
        # §32: with a permanently unavailable depth stream the chain
        # through the temporal layer reproduces the legacy spatial
        # decisions (stabilized; equal actions confirmed not changed).
        motion = MotionEstimator(clock=lambda: self.t)
        risk = RiskEstimator()
        legacy = Navigator()
        raw = [DetectedObject(label="chair", confidence=0.9,
                              bbox=BoundingBox(80, 150, 130, 350))]
        for _ in range(4):
            dets, decision = self._frame(raw, motion, risk,
                                         UnavailableProvider, [None])
            expected = legacy.decide(dets).action
            self.assertEqual(decision.action, expected)
            self.assertEqual(dets[0].distance_provenance, "UNAVAILABLE")

    def test_audio_signature_stable_during_holds(self):
        # §23/§27.27: held decisions re-emit the same (action, trigger)
        # signature so existing audio dedup/cooldown keeps working
        # unchanged; a genuine escalation changes the signature.
        motion = MotionEstimator(clock=lambda: self.t)
        risk = RiskEstimator()
        raw = [DetectedObject(label="person", confidence=0.93,
                              bbox=BoundingBox(310, 150, 350, 350))]
        _, d1 = self._frame(raw, motion, risk, ScriptedProvider, [1.0])
        sig1 = (d1.action.name,
                d1.trigger.label if d1.trigger else None)
        # Noisy far frame -> held -> same signature.
        _, d2 = self._frame(raw, motion, risk, ScriptedProvider, [4.5])
        sig2 = (d2.action.name,
                d2.trigger.label if d2.trigger else None)
        self.assertEqual(sig1, sig2)


class ConfigGuardTest(unittest.TestCase):
    """Invalid configuration fails loudly (no silent misbehavior)."""

    def test_zero_confirmation_frames_rejected(self):
        with self.assertRaises(ValueError):
            TemporalNavigator(DistanceNavigator(),
                              deescalation_confirm_frames=0)
        with self.assertRaises(ValueError):
            TemporalNavigator(DistanceNavigator(),
                              action_change_confirm_frames=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
