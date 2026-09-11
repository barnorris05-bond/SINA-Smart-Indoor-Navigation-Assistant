"""
tests/test_object_tracker.py

PHASE 5 TRACKING TESTS — hardware-free by construction.

No OAK-D, no DepthAI, no USB, no GPU, no network. Pure logic tests on
DetectedObject streams.

Lifecycle semantics used throughout (documented choice):
  - A detection that creates a track yields state NEW on its first
    update and ACTIVE from the second consecutive match onward.
  - A matched track that is absent in a later frame becomes MISSING
    (retained up to MAX_MISSED_FRAMES) and returns to ACTIVE on rematch.
  - A track missing for more than MAX_MISSED_FRAMES consecutive frames
    is dropped (EXPIRED) at the next update; IDs are never reused.
  - reset() clears tracks; ID numbering stays monotonic afterwards
    (chosen policy: no reuse, simpler logs/tests).

Run:
    ./.venv/Scripts/python.exe -m pytest tests/test_object_tracker.py -q
"""

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.object_detector import DetectedObject, BoundingBox
from vision.detection_manager import DetectionManager
from vision.object_tracker import ObjectTracker, TrackState, iou, centroid_distance
from vision.distance_fusion import DistanceFusion
from depth.mock_provider import ScenarioDepthProvider
from depth.provider import DepthMeasurement, DistanceProvenance
from navigation.distance_navigator import DistanceNavigator
from navigation.navigation_types import NavigationAction
from config.navigation import OBJECT_PRIORITIES


def make_det(label, bbox, confidence=0.9, distance=None,
             provenance="UNAVAILABLE", source=None, track_id=None):
    """Raw detection as YOLODetector would emit it (pre-enrichment)."""
    return DetectedObject(
        label=label,
        confidence=confidence,
        bbox=BoundingBox(*bbox),
        distance=distance,
        distance_provenance=provenance,
        distance_source=source,
        track_id=track_id,
    )


def person(bbox, **kw):
    return make_det("person", bbox, **kw)


# ==============================================================
# Basics: creation, persistence, movement
# ==============================================================

class TestBasics(unittest.TestCase):
    def setUp(self):
        self.tracker = ObjectTracker()

    def test_1_new_object_gets_first_id(self):
        dets = [person((100, 100, 200, 300))]
        out = self.tracker.update(dets)
        self.assertEqual(out[0].track_id, 1)
        track = self.tracker.active_tracks()[0]
        self.assertEqual(track.state, TrackState.NEW)  # first update = NEW
        self.assertEqual(track.hits, 1)

    def test_1b_second_consecutive_match_is_active(self):
        self.tracker.update([person((100, 100, 200, 300))])
        self.tracker.update([person((100, 100, 200, 300))])
        track = self.tracker.active_tracks()[0]
        self.assertEqual(track.state, TrackState.ACTIVE)

    def test_2_same_object_next_frame_keeps_id(self):
        d1 = self.tracker.update([person((100, 100, 200, 300))])[0]
        d2 = self.tracker.update([person((105, 102, 205, 302))])[0]
        self.assertEqual(d1.track_id, 1)
        self.assertEqual(d2.track_id, 1)

    def test_3_meaningful_movement_keeps_id(self):
        # Center moves (200,200) -> (240,205) -> (280,210): within the
        # configured centroid limit every frame -> same track.
        ids = []
        for x in (150, 190, 230):
            det = self.tracker.update([person((x, 100, x + 100, 300))])[0]
            ids.append(det.track_id)
        self.assertEqual(ids, [1, 1, 1])

    def test_no_id_reuse_after_expiry(self):
        # Expiry semantics: a track missing for more than
        # MAX_MISSED_FRAMES consecutive frames is dropped at the next
        # update. Small tracker for a fast test.
        t = ObjectTracker(max_missed_frames=2)
        t.update([person((100, 100, 200, 300))])   # track 1
        for _ in range(4):                          # well beyond max
            t.update([])
        self.assertEqual(t.track_count(), 0)        # expired/removed
        d = t.update([person((100, 100, 200, 300))])[0]
        self.assertEqual(d.track_id, 2)             # monotonic, no reuse


# ==============================================================
# Disappearance / lifecycle
# ==============================================================

class TestLifecycle(unittest.TestCase):
    def test_4_temporary_disappearance_keeps_id(self):
        t = ObjectTracker()
        d1 = t.update([person((100, 100, 200, 300))])[0]
        t.update([])                       # MISSING
        track = t.active_tracks()[0]
        self.assertEqual(track.state, TrackState.MISSING)
        self.assertEqual(track.missed_frames, 1)
        d3 = t.update([person((100, 100, 200, 300))])[0]   # returns
        self.assertEqual(d3.track_id, d1.track_id)
        self.assertEqual(t.active_tracks()[0].state, TrackState.ACTIVE)

    def test_missing_survives_up_to_limit(self):
        t = ObjectTracker(max_missed_frames=3)
        t.update([person((100, 100, 200, 300))])
        for _ in range(3):
            t.update([])
        self.assertEqual(t.track_count(), 1)   # still retained at limit

    def test_5_beyond_limit_expires_and_new_id(self):
        t = ObjectTracker(max_missed_frames=2)
        d1 = t.update([person((100, 100, 200, 300))])[0]
        for _ in range(10):                     # far beyond the limit
            t.update([])
        self.assertEqual(t.track_count(), 0)
        d2 = t.update([person((100, 100, 200, 300))])[0]
        self.assertNotEqual(d1.track_id, d2.track_id)


# ==============================================================
# Multi-object matching
# ==============================================================

class TestMultiObject(unittest.TestCase):
    def setUp(self):
        self.tracker = ObjectTracker()

    def test_6_two_same_class_objects_get_distinct_ids(self):
        a, b = self.tracker.update([
            person((50, 100, 150, 300)),
            person((450, 100, 550, 300)),
        ])
        self.assertNotEqual(a.track_id, b.track_id)
        self.assertEqual({a.track_id, b.track_id}, {1, 2})

    def test_7_detection_order_swap_preserves_ids(self):
        a1, b1 = self.tracker.update([
            person((50, 100, 150, 300)),     # A left
            person((450, 100, 550, 300)),    # B right
        ])
        # Next frame: detector returns [B, A] (reversed order).
        b2, a2 = self.tracker.update([
            person((450, 100, 550, 300)),
            person((50, 100, 150, 300)),
        ])
        self.assertEqual(a2.track_id, a1.track_id)
        self.assertEqual(b2.track_id, b1.track_id)
        # And back again.
        a3, b3 = self.tracker.update([
            person((50, 100, 150, 300)),
            person((450, 100, 550, 300)),
        ])
        self.assertEqual(a3.track_id, a1.track_id)
        self.assertEqual(b3.track_id, b1.track_id)

    def test_8_different_labels_never_merge(self):
        p, c = self.tracker.update([
            person((200, 100, 300, 300)),
            make_det("chair", (200, 100, 300, 300)),   # identical box
        ])
        self.assertNotEqual(p.track_id, c.track_id)

    def test_one_to_one_matching_duplicate_detections(self):
        # One track exists; TWO identical detections appear. Exactly one
        # may inherit the track; the other must get a fresh ID.
        self.tracker.update([person((100, 100, 200, 300))])   # track 1
        a, b = self.tracker.update([
            person((100, 100, 200, 300)),
            person((100, 100, 200, 300)),
        ])
        ids = {a.track_id, b.track_id}
        self.assertEqual(len(ids), 2)          # no shared identity
        self.assertIn(1, ids)                  # one kept track 1
        self.assertIn(2, ids)                  # other became a new track

    def test_9_crossing_objects_deterministic(self):
        # Two persons approach and cross. Geometry-only tracking cannot
        # guarantee identity THROUGH the crossing instant (documented
        # limitation, no motion model) — the contract tested here is
        # DETERMINISM: identical input streams produce identical
        # assignments, and no duplicate/lost IDs occur.
        def run():
            t = ObjectTracker()
            history = []
            for i in range(5):
                x_a = 100 + i * 60             # 100 -> 340
                x_b = 520 - i * 60             # 520 -> 280
                dets = t.update([
                    person((x_a, 100, x_a + 80, 300)),
                    person((x_b, 100, x_b + 80, 300)),
                ])
                history.append(sorted(d.track_id for d in dets))
            return history

        h1, h2 = run(), run()
        self.assertEqual(h1, h2)               # deterministic
        for pair in h1:
            self.assertEqual(len(pair), 2)
            self.assertNotEqual(pair[0], pair[1])


# ==============================================================
# Reset
# ==============================================================

class TestReset(unittest.TestCase):
    def test_10_reset_clears_tracks_and_keeps_ids_monotonic(self):
        t = ObjectTracker()
        t.update([person((100, 100, 200, 300))])   # 1
        t.update([person((450, 100, 550, 300))])   # 2
        self.assertEqual(t.track_count(), 2)

        t.reset()
        self.assertEqual(t.track_count(), 0)

        d = t.update([person((100, 100, 200, 300))])[0]
        # Chosen policy: IDs continue monotonically after reset (no
        # reuse) — documented in ObjectTracker.reset().
        self.assertEqual(d.track_id, 3)

    def test_reset_rejects_malformed_construction(self):
        with self.assertRaises(ValueError):
            ObjectTracker(max_missed_frames=-1)
        with self.assertRaises(ValueError):
            ObjectTracker(min_iou=1.5)
        with self.assertRaises(ValueError):
            ObjectTracker(max_centroid_distance=-5)


# ==============================================================
# Data integrity: distance / provenance untouched
# ==============================================================

class TestDataIntegrity(unittest.TestCase):
    def setUp(self):
        self.tracker = ObjectTracker()

    def _run_two_frames(self, **kw):
        d1 = self.tracker.update([person((100, 100, 200, 300), **kw)])[0]
        d2 = self.tracker.update([person((102, 101, 202, 301), **kw)])[0]
        return d1, d2

    def test_11_distance_preserved(self):
        d1, d2 = self._run_two_frames(distance=2.8, provenance="SIMULATED")
        self.assertEqual(d1.distance, 2.8)
        self.assertEqual(d2.distance, 2.8)

    def test_12_provenance_and_source_preserved(self):
        d1, d2 = self._run_two_frames(
            distance=2.5, provenance="SIMULATED", source="fixed"
        )
        for det in (d1, d2):
            self.assertEqual(det.distance_provenance, "SIMULATED")
            self.assertEqual(det.distance_source, "fixed")

    def test_13_unavailable_distance_not_invented(self):
        d1, d2 = self._run_two_frames(
            distance=None, provenance="UNAVAILABLE"
        )
        self.assertIsNone(d1.distance)
        self.assertIsNone(d2.distance)
        self.assertEqual(d1.distance_provenance, "UNAVAILABLE")

    def test_14_stale_provenance_preserved_not_promoted(self):
        d1, d2 = self._run_two_frames(
            distance=0.5, provenance="STALE"
        )
        for det in (d1, d2):
            self.assertEqual(det.distance_provenance, "STALE")
            self.assertEqual(det.distance, 0.5)   # value untouched, honestly labeled

    def test_core_fields_preserved(self):
        det = make_det("person", (100, 100, 200, 300), confidence=0.77,
                       distance=3.3, provenance="SIMULATED", source="s")
        out = self.tracker.update([det])[0]
        self.assertIs(out, det)                       # same instance
        self.assertEqual(out.label, "person")
        self.assertAlmostEqual(out.confidence, 0.77)
        self.assertEqual((out.bbox.x1, out.bbox.y1, out.bbox.x2, out.bbox.y2),
                         (100, 100, 200, 300))


# ==============================================================
# Pipeline integration
# ==============================================================

class TestPipelineIntegration(unittest.TestCase):
    def test_15_detection_manager_integration(self):
        manager = DetectionManager()
        tracker = ObjectTracker()

        raw = [person((300, 150, 360, 350), distance=2.5,
                      provenance="SIMULATED", source="fixed")]
        dets = manager.process(raw, frame_width=640)
        tracked = tracker.update(dets)

        det = tracked[0]
        self.assertEqual(det.label, "person")
        self.assertEqual(det.region, "CENTER")
        self.assertEqual(det.priority, 100)
        self.assertEqual((det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2),
                         (300, 150, 360, 350))
        self.assertEqual(det.track_id, 1)
        self.assertEqual(det.distance_provenance, "SIMULATED")
        self.assertEqual(det.distance_source, "fixed")

    def test_16_fusion_integration_preserves_track_id(self):
        manager = DetectionManager()
        tracker = ObjectTracker()
        fusion = DistanceFusion(ScenarioDepthProvider(), clock=lambda: 1000.0)

        raw = [person((300, 150, 360, 350))]
        dets = manager.process(raw, frame_width=640)
        tracked = tracker.update(dets)
        fusion.update()
        fusion.fuse(tracked, (400, 640))

        det = tracked[0]
        self.assertEqual(det.track_id, 1)                     # survived fusion
        self.assertAlmostEqual(det.distance, 1.2)             # mock table
        self.assertEqual(det.distance_provenance, "SIMULATED")
        self.assertEqual(det.distance_source, "mock_scenario")

    def test_17_navigation_decisions_unchanged_by_tracking(self):
        """Tracking adds IDs only; decisions must be identical with and
        without the tracker in the chain."""
        clock = lambda: 1000.0
        raw = [person((300, 150, 360, 350))]

        # Without tracker:
        base_dets = DetectionManager().process(
            [person((300, 150, 360, 350))], frame_width=640)
        DistanceFusion(ScenarioDepthProvider(), clock=clock).fuse(
            base_dets, (400, 640))
        base = DistanceNavigator().decide(base_dets)

        # With tracker:
        tracked_dets = ObjectTracker().update(
            DetectionManager().process(
                [person((300, 150, 360, 350))], frame_width=640))
        DistanceFusion(ScenarioDepthProvider(), clock=clock).fuse(
            tracked_dets, (400, 640))
        with_tracking = DistanceNavigator().decide(tracked_dets)

        self.assertEqual(base.action, with_tracking.action)
        self.assertEqual(base.reason, with_tracking.reason)

    def test_legacy_fallback_unaffected(self):
        # No usable distance anywhere -> legacy spatial policy, tracking
        # or not (requirement 43).
        dets = ObjectTracker().update(
            DetectionManager().process(
                [make_det("rocket", (300, 150, 360, 350))], frame_width=640))
        decision = DistanceNavigator().decide(dets)
        self.assertEqual(decision.action, NavigationAction.SLOW_DOWN)


# ==============================================================
# End-to-end mock: approaching person (requirement 30)
# ==============================================================

class TestApproachingPersonE2E(unittest.TestCase):
    def test_track_stable_and_navigation_escalates(self):
        T = 1000.0
        distances = [2.8, 2.4, 1.5, 1.0]
        expected_actions = [
            NavigationAction.CONTINUE,    # 2.8 > 2.5 (dev thresholds)
            NavigationAction.SLOW_DOWN,   # 2.4 <= 2.5
            NavigationAction.SLOW_DOWN,   # 1.5 <= 2.5
            NavigationAction.STOP,        # 1.0 <= 1.2
        ]

        class ScriptedProvider:
            """SIMULATED provider with a scripted per-frame distance."""

            def __init__(self, seq):
                self._seq = list(seq)
                self._i = 0

            @property
            def name(self):
                return "scripted"

            def start(self):
                pass

            def stop(self):
                pass

            def update(self):
                pass

            def get_measurement(self, bbox, frame_shape, label=None, now=None):
                d = self._seq[min(self._i, len(self._seq) - 1)]
                self._i += 1
                return DepthMeasurement(d, DistanceProvenance.SIMULATED,
                                        "fixed", None, T)

        tracker = ObjectTracker()
        fusion = DistanceFusion(ScriptedProvider(distances), clock=lambda: T)
        navigator = DistanceNavigator()
        manager = DetectionManager()

        ids = []
        provenance = []
        actions = []
        for i, expected_distance in enumerate(distances):
            # Slight approach drift, same identity throughout.
            raw = [person((300 + i * 2, 150, 360 + i * 2, 350))]
            dets = manager.process(raw, frame_width=640)
            tracked = tracker.update(dets)
            fusion.update()
            fusion.fuse(tracked, (400, 640))
            decision = navigator.decide(tracked)

            ids.append(tracked[0].track_id)
            provenance.append(tracked[0].distance_provenance)
            actions.append(decision.action)
            self.assertAlmostEqual(tracked[0].distance, expected_distance)

        self.assertEqual(ids, [1, 1, 1, 1])                      # persistent
        self.assertEqual(provenance, ["SIMULATED"] * 4)          # honest
        self.assertEqual(actions, expected_actions)              # thresholds unchanged


# ==============================================================
# Geometry helpers
# ==============================================================

class TestGeometry(unittest.TestCase):
    def test_iou_identical_boxes(self):
        a = BoundingBox(0, 0, 10, 10)
        self.assertAlmostEqual(iou(a, a), 1.0)

    def test_iou_disjoint(self):
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(20, 20, 30, 30)
        self.assertEqual(iou(a, b), 0.0)

    def test_iou_half_overlap(self):
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(5, 0, 15, 10)
        self.assertAlmostEqual(iou(a, b), 50.0 / 150.0)

    def test_centroid_distance_symmetric(self):
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(10, 0, 20, 10)
        self.assertAlmostEqual(centroid_distance(a, b), 10.0)
        self.assertAlmostEqual(centroid_distance(b, a), 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
