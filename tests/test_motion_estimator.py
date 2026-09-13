"""
tests/test_motion_estimator.py

PHASE 6 — MOTION / APPROACH ESTIMATION TESTS (hardware-free).

Covers (master roadmap 4.7):
  approaching / stationary / receding / insufficient history /
  missing distance / STALE / UNAVAILABLE / SIMULATED / multiple tracks /
  track expiration (purge) / small measurement noise (no oscillation)

Plus:
  - hysteresis dead zones (no flicker, no oscillation)
  - detection-gap safety (no pre-gap rate blending)
  - provenance honesty (MEASURED only if ALL samples measured; mixed
    -> SIMULATED; simulated speed never claims to be real)
  - additive-only stamping (distance/track_id/label/priority untouched)
  - pipeline integration:
      DetectionManager -> ObjectTracker -> DistanceFusion -> MotionEstimator

No OAK-D, no DepthAI, no OpenCV windows, no pyttsx3.

Run:
    ./.venv/Scripts/python.exe tests/test_motion_estimator.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.object_detector import DetectedObject, BoundingBox
from vision.detection_manager import DetectionManager
from vision.object_tracker import ObjectTracker
from vision.motion_estimator import MotionEstimator, MotionState
from vision.distance_fusion import DistanceFusion
from depth.provider import DepthProvider, DepthMeasurement, DistanceProvenance
from navigation.distance_navigator import DistanceNavigator
from navigation.navigation_types import NavigationAction


FRAME_W, FRAME_H = 640, 400


def make_det(track_id, distance=None, provenance="SIMULATED",
             label="person", distance_source="test"):
    return DetectedObject(
        label=label,
        confidence=0.9,
        bbox=BoundingBox(300, 150, 340, 350),
        track_id=track_id,
        distance=distance,
        distance_provenance=provenance,
        distance_source=distance_source,
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


class MotionEstimatorTest(unittest.TestCase):
    """Unit tests with an injected virtual clock."""

    def setUp(self):
        self.t = 1000.0
        self.est = MotionEstimator(clock=lambda: self.t)

    def _tick(self, *dets, dt=0.25):
        """Feed one frame; advance the virtual clock; return detections."""
        out = self.est.update(list(dets))
        self.t += dt
        return out

    # ------------------------------------------------------------
    # Core states
    # ------------------------------------------------------------

    def test_new_track_insufficient_history_is_unknown(self):
        det = make_det(1, 3.0)
        out = self._tick(det)[0]
        self.assertEqual(out.motion_state, "UNKNOWN")
        self.assertIsNone(out.closing_rate_mps)
        self.assertEqual(out.motion_provenance, "UNAVAILABLE")

    def test_short_span_still_unknown(self):
        # 3 samples, span exactly MIN_OBSERVATION_S: classification is
        # allowed from the boundary (span is not below the minimum).
        det = make_det(1, 3.0)
        self._tick(det)                           # t0
        self._tick(make_det(1, 2.9))              # +0.25 s
        out = self._tick(make_det(1, 2.8))[0]     # +0.25 s -> span 0.5
        self.assertEqual(out.motion_state, "APPROACHING")
        # 0.1 m per 0.25 s step = 0.4 m/s closing rate.
        self.assertAlmostEqual(out.closing_rate_mps, 0.4, places=6)

    def test_approaching(self):
        d = [3.0, 2.8, 2.6, 2.4, 2.2]      # 0.2 m per 0.25 s -> 0.8 m/s
        outs = [self._tick(make_det(1, dist))[0] for dist in d]
        final = outs[-1]
        self.assertEqual(final.motion_state, "APPROACHING")
        self.assertAlmostEqual(final.closing_rate_mps, 0.8, places=6)

    def test_stationary(self):
        d = [2.5, 2.502, 2.498, 2.501, 2.499]  # mm-level jitter
        outs = [self._tick(make_det(1, dist))[0] for dist in d]
        final = outs[-1]
        self.assertEqual(final.motion_state, "STATIONARY")
        self.assertLess(abs(final.closing_rate_mps), 0.05)

    def test_receding(self):
        d = [2.0, 2.2, 2.4, 2.6, 2.8]      # 0.8 m/s receding
        outs = [self._tick(make_det(1, dist))[0] for dist in d]
        final = outs[-1]
        self.assertEqual(final.motion_state, "RECEDING")
        self.assertAlmostEqual(final.closing_rate_mps, -0.8, places=6)

    # ------------------------------------------------------------
    # Hysteresis / noise robustness
    # ------------------------------------------------------------

    def test_noise_does_not_flip_stationary(self):
        # Alternating jitter around a stationary object: must never
        # oscillate into APPROACHING/RECEDING.
        seq = [2.5, 2.52, 2.48, 2.51, 2.49, 2.505, 2.495, 2.5]
        states = [self._tick(make_det(1, dist))[0].motion_state
                  for dist in seq]
        self.assertTrue(all(s in ("UNKNOWN", "STATIONARY") for s in states),
                        f"oscillation detected: {states}")

    def test_hysteresis_dead_zone_keeps_previous_state(self):
        # Establish APPROACHING strongly...
        for dist in [3.0, 2.8, 2.6, 2.4, 2.2]:
            self._tick(make_det(1, dist))
        # ...then hold a CONSTANT dead-zone rate (0.05 < 0.1 < 0.15 m/s):
        # 0.1 m per 0.5 s step. Hysteresis must keep APPROACHING while
        # the rate sits in the dead zone between the two bands.
        for dist in [2.1, 2.0, 1.9, 1.8]:
            (out,) = self._tick(make_det(1, dist), dt=0.5)
            self.assertEqual(
                out.motion_state, "APPROACHING",
                f"flickered to {out.motion_state} at rate "
                f"{out.closing_rate_mps:.3f}")

    def test_state_eventually_updates_after_sustained_change(self):
        # Hysteresis must not mean "stuck forever": sustained receding
        # beyond the dead band reclassifies to RECEDING.
        for dist in [3.0, 2.8, 2.6, 2.4, 2.2]:   # approach
            self._tick(make_det(1, dist))
        for dist in [2.4, 2.6, 2.8, 3.0, 3.2]:   # sustained recede
            (out,) = self._tick(make_det(1, dist))
        self.assertEqual(out.motion_state, "RECEDING")

    # ------------------------------------------------------------
    # Sample validity / provenance
    # ------------------------------------------------------------

    def test_missing_distance_never_enters_history(self):
        det = make_det(1, None, provenance="UNAVAILABLE")
        for _ in range(5):
            (out,) = self._tick(det)
        self.assertEqual(out.motion_state, "UNKNOWN")
        self.assertEqual(self.est.history_len(1), 0)

    def test_stale_distance_rejected(self):
        det = make_det(1, 1.5, provenance="STALE")
        for _ in range(5):
            (out,) = self._tick(det)
        self.assertEqual(out.motion_state, "UNKNOWN")
        self.assertEqual(self.est.history_len(1), 0)

    def test_valid_stream_not_poisoned_by_stale_frames(self):
        # Valid samples, then STALE frames, then valid again: stale
        # frames must contribute nothing and must not reset history.
        for dist in [3.0, 2.8, 2.6]:
            self._tick(make_det(1, dist))
        for _ in range(3):
            self._tick(make_det(1, 99.0, provenance="STALE"))
        self.assertEqual(self.est.history_len(1), 3)
        (out,) = self._tick(make_det(1, 2.4))
        self.assertEqual(out.motion_state, "APPROACHING")
        self.assertEqual(self.est.history_len(1), 4)

    def test_unavailable_amid_valid_frames_does_not_reset(self):
        for dist in [3.0, 2.8, 2.6]:
            self._tick(make_det(1, dist))
        (out,) = self._tick(make_det(1, None, provenance="UNAVAILABLE"))
        self.assertEqual(self.est.history_len(1), 3)
        (out,) = self._tick(make_det(1, 2.4))
        self.assertEqual(out.motion_state, "APPROACHING")

    def test_simulated_provenance_stays_simulated(self):
        for dist in [3.0, 2.8, 2.6, 2.4]:
            (out,) = self._tick(make_det(1, dist, provenance="SIMULATED"))
        self.assertEqual(out.motion_provenance, "SIMULATED")

    def test_all_measured_samples_give_measured_motion(self):
        for dist in [3.0, 2.8, 2.6, 2.4]:
            (out,) = self._tick(make_det(1, dist, provenance="MEASURED"))
        self.assertEqual(out.motion_state, "APPROACHING")
        self.assertEqual(out.motion_provenance, "MEASURED")

    def test_mixed_provenance_is_simulated_not_measured(self):
        # One SIMULATED sample anywhere in the window -> the estimate
        # must be SIMULATED (simulated speed never claims to be real).
        self._tick(make_det(1, 3.0, provenance="MEASURED"))
        self._tick(make_det(1, 2.8, provenance="SIMULATED"))
        (out,) = self._tick(make_det(1, 2.6, provenance="MEASURED"))
        self.assertEqual(out.motion_state, "APPROACHING")
        self.assertEqual(out.motion_provenance, "SIMULATED")

    def test_no_track_id_is_honest_unknown(self):
        det = make_det(None, 3.0)
        (out,) = self._tick(det)
        self.assertEqual(out.motion_state, "UNKNOWN")
        self.assertEqual(out.motion_provenance, "UNAVAILABLE")

    # ------------------------------------------------------------
    # Multi-track, gaps, expiry, reset
    # ------------------------------------------------------------

    def test_multiple_tracks_classified_independently(self):
        a = [3.0, 2.8, 2.6, 2.4, 2.2]      # approaching
        b = [2.0, 2.2, 2.4, 2.6, 2.8]      # receding
        outs = None
        for da, db in zip(a, b):
            outs = self._tick(make_det(1, da), make_det(2, db))
        self.assertEqual(outs[0].motion_state, "APPROACHING")
        self.assertEqual(outs[1].motion_state, "RECEDING")

    def test_gap_does_not_blend_pre_gap_rate(self):
        # Strong approach, then a 10 s detection gap: after the gap the
        # old samples are outside WINDOW_SPAN_S, so motion must be
        # UNKNOWN until fresh evidence re-accumulates - never a rate
        # computed across the gap.
        for dist in [3.0, 2.8, 2.6, 2.4, 2.2]:
            self._tick(make_det(1, dist))
        self.t += 10.0                      # detection/depth outage
        (out,) = self._tick(make_det(1, 2.2))
        self.assertEqual(out.motion_state, "UNKNOWN")
        self.assertIsNone(out.closing_rate_mps)
        # Fresh evidence re-accumulates -> honest classification again.
        (out,) = self._tick(make_det(1, 2.0))
        (out,) = self._tick(make_det(1, 1.8))
        (out,) = self._tick(make_det(1, 1.6))
        self.assertEqual(out.motion_state, "APPROACHING")

    def test_history_purged_after_track_expiration(self):
        for dist in [3.0, 2.8, 2.6]:
            self._tick(make_det(1, dist))
        # Absent for more than MAX_HISTORY_AGE_FRAMES (15) frames.
        for _ in range(16):
            self._tick()
        self.assertEqual(self.est.history_len(1), 0)
        # Same track_id reappearing starts clean (no stale identity).
        (out,) = self._tick(make_det(1, 3.0))
        self.assertEqual(out.motion_state, "UNKNOWN")

    def test_reset_clears_everything(self):
        for dist in [3.0, 2.8, 2.6]:
            self._tick(make_det(1, dist))
        self.est.reset()
        (out,) = self._tick(make_det(1, 3.0))
        self.assertEqual(out.motion_state, "UNKNOWN")
        self.assertEqual(self.est.history_len(1), 1)

    # ------------------------------------------------------------
    # Additive-only guarantee
    # ------------------------------------------------------------

    def test_estimator_does_not_modify_existing_fields(self):
        det = make_det(1, 2.5)
        det.priority = 100
        det.region = "CENTER"
        before = (det.distance, det.distance_provenance, det.distance_source,
                  det.track_id, det.label, det.confidence, det.priority,
                  det.region, det.bbox)
        for dist in [2.5, 2.5, 2.5, 2.5]:
            (out,) = self._tick(det)
        after = (out.distance, out.distance_provenance, out.distance_source,
                 out.track_id, out.label, out.confidence, out.priority,
                 out.region, out.bbox)
        self.assertEqual(before, after)
        # New fields stamped:
        self.assertEqual(out.motion_state, "STATIONARY")
        self.assertEqual(out.motion_provenance, "SIMULATED")


class MotionPipelineIntegrationTest(unittest.TestCase):
    """
    Full chain with real SINA classes (no hardware):

        DetectionManager -> ObjectTracker -> DistanceFusion ->
        MotionEstimator -> DistanceNavigator (semantics unchanged)
    """

    def setUp(self):
        self.t = 2000.0
        self.manager = DetectionManager()
        self.tracker = ObjectTracker()
        self.navigator = DistanceNavigator()

    def _fused_tracked(self, distances, dt=0.25):
        """One pipeline pass with scripted SIMULATED distances."""
        self.t += dt
        clock = lambda: self.t
        fusion = DistanceFusion(
            ScriptedProvider(distances), clock=clock)
        raw = [DetectedObject(label="person", confidence=0.93,
                              bbox=BoundingBox(310, 150, 350, 350))]
        dets = self.manager.process(raw, FRAME_W)
        dets = self.tracker.update(dets)
        fusion.fuse(dets, (FRAME_H, FRAME_W))
        return dets

    def test_track_id_and_motion_through_full_pipeline(self):
        seq = [3.0, 2.8, 2.6, 2.4, 2.2, 2.0]
        est = MotionEstimator(clock=lambda: self.t)
        last = None
        for d in seq:
            dets = self._fused_tracked([d])
            last = est.update(dets)
        det = last[0]
        # Persistent identity through the whole chain:
        self.assertEqual(det.track_id, 1)
        # Mock distances stayed SIMULATED; motion honesty follows:
        self.assertEqual(det.distance_provenance, "SIMULATED")
        self.assertEqual(det.motion_provenance, "SIMULATED")
        self.assertEqual(det.motion_state, "APPROACHING")
        self.assertGreater(det.closing_rate_mps, 0.5)

    def test_navigation_decisions_unchanged_by_motion_layer(self):
        # Same decision with and without the estimator in the chain:
        # motion is annotation, not policy.
        dets_no_motion = self._fused_tracked([1.0])
        decision_without = self.navigator.decide(
            [type(dets_no_motion[0])(**{
                "label": dets_no_motion[0].label,
                "confidence": dets_no_motion[0].confidence,
                "bbox": dets_no_motion[0].bbox,
                "distance": dets_no_motion[0].distance,
                "distance_provenance": dets_no_motion[0].distance_provenance,
                "distance_source": dets_no_motion[0].distance_source,
            })]
        )
        dets = self._fused_tracked([0.9])
        est = MotionEstimator(clock=lambda: self.t)
        dets = est.update(dets)
        decision_with = self.navigator.decide(dets)
        self.assertEqual(decision_without.action, NavigationAction.STOP)
        self.assertEqual(decision_with.action, NavigationAction.STOP)
        # Motion fields present on the annotated path:
        self.assertIsNotNone(dets[0].motion_state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
