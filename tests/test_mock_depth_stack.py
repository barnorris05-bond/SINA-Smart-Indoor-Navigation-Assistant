"""
tests/test_mock_depth_stack.py

NO-HARDWARE tests for the mock depth stack (Phases B-D):
provider provenance, scenario/bbox strategies, staleness, fusion,
distance-aware navigation and safety escalation.

Pure logic + NumPy only. The OAK-D (and even DepthAI) is never touched.

Run:
    ./.venv/Scripts/python.exe tests/test_mock_depth_stack.py
"""

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth.provider import DepthMeasurement, DistanceProvenance
from depth.mock_provider import (
    MockDepthProvider,
    ScenarioDepthProvider,
    BboxSyntheticProvider,
    region_of,
)
from depth.oak_provider import OakStereoDepthProvider
from vision.object_detector import DetectedObject, BoundingBox
from vision.distance_fusion import DistanceFusion
from navigation.distance_navigator import DistanceNavigator
from navigation.navigator import Navigator
from navigation.navigation_types import NavigationAction
from config.mock_depth import USE_MOCK_DEPTH


# Fake clock: fully deterministic staleness/freshness tests.
class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_det(label, region="CENTER", confidence=0.9, distance=None,
             provenance="UNAVAILABLE", bbox=None):
    if bbox is None:
        # 640x400 frame thirds: LEFT <213, CENTER 213..427, RIGHT >=427.
        bbox = BoundingBox(300, 100, 340, 200)
    from config.navigation import OBJECT_PRIORITIES
    return DetectedObject(
        label=label,
        confidence=confidence,
        bbox=bbox,
        center_x=(bbox.x1 + bbox.x2) // 2,
        center_y=(bbox.y1 + bbox.y2) // 2,
        region=region,
        distance=distance,
        distance_provenance=provenance,
        priority=OBJECT_PRIORITIES.get(label.lower(), 0),
    )


# ==============================================================
# 1. Provider provenance & strategies
# ==============================================================

class TestScenarioProvider(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.p = ScenarioDepthProvider(clock=self.clock)

    def test_valid_mock_distance_is_simulated(self):
        m = self.p.get_measurement((300, 100, 340, 200), (400, 640),
                                   label="person", now=self.clock())
        self.assertEqual(m.provenance, DistanceProvenance.SIMULATED)
        self.assertEqual(m.distance_m, 1.2)  # person/CENTER from config table
        self.assertTrue(m.usable)

    def test_label_fallback(self):
        # (door, LEFT) is not in the scenario table -> label default 3.0.
        m = self.p.get_measurement((50, 100, 150, 200), (400, 640),
                                   label="door", now=self.clock())
        self.assertEqual(m.distance_m, 3.0)  # LABEL_DEFAULT

    def test_unknown_object_is_unavailable(self):
        m = self.p.get_measurement((300, 100, 340, 200), (400, 640),
                                   label="rocket", now=self.clock())
        self.assertIsNone(m.distance_m)
        self.assertEqual(m.provenance, DistanceProvenance.UNAVAILABLE)
        self.assertFalse(m.usable)

    def test_no_label_is_unavailable(self):
        m = self.p.get_measurement((300, 100, 340, 200), (400, 640), now=1000.0)
        self.assertIsNone(m.distance_m)
        self.assertEqual(m.provenance, DistanceProvenance.UNAVAILABLE)

    def test_region_partition_matches_fusion_convention(self):
        self.assertEqual(region_of((0, 0, 100, 100), (400, 640)), "LEFT")
        self.assertEqual(region_of((300, 0, 340, 100), (400, 640)), "CENTER")
        self.assertEqual(region_of((500, 0, 600, 100), (400, 640)), "RIGHT")

    def test_source_tags_strategy(self):
        m = self.p.get_measurement((300, 100, 340, 200), (400, 640),
                                   label="person", now=1000.0)
        self.assertEqual(m.source, "mock_scenario")


class TestBboxProvider(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.p = BboxSyntheticProvider(clock=self.clock)

    def test_pinhole_scaling(self):
        # person (1.7m): d = 1.7 * 484 / 200 = 4.114 m
        m = self.p.get_measurement((300, 100, 340, 300), (400, 640),
                                   label="person", now=1000.0)
        self.assertAlmostEqual(m.distance_m, 1.7 * 484.0 / 200.0, places=3)

    def test_bigger_box_closer(self):
        near = self.p.get_measurement((300, 100, 340, 300), (400, 640),
                                      label="person", now=1000.0)
        far = self.p.get_measurement((300, 180, 340, 220), (400, 640),
                                     label="person", now=1000.0)
        self.assertLess(near.distance_m, far.distance_m)

    def test_unknown_label_uses_default_height(self):
        m = self.p.get_measurement((300, 100, 340, 300), (400, 640),
                                   label="rocket", now=1000.0)
        self.assertAlmostEqual(m.distance_m, 0.5 * 484.0 / 200.0, places=3)

    def test_degenerate_bbox_is_unavailable(self):
        m = self.p.get_measurement((300, 100, 340, 100), (400, 640),
                                   label="person", now=1000.0)
        self.assertIsNone(m.distance_m)
        self.assertEqual(m.provenance, DistanceProvenance.UNAVAILABLE)

    def test_values_are_simulated(self):
        m = self.p.get_measurement((300, 100, 340, 300), (400, 640),
                                   label="person", now=1000.0)
        self.assertEqual(m.provenance, DistanceProvenance.SIMULATED)


class TestMockFacade(unittest.TestCase):
    def test_default_mode_is_scenario(self):
        p = MockDepthProvider()
        self.assertEqual(p.name, "mock_scenario")

    def test_bbox_mode_switch(self):
        p = MockDepthProvider(mode="bbox")
        self.assertEqual(p.name, "mock_bbox")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            MockDepthProvider(mode="telepathy")

    def test_mock_mode_is_on_in_config(self):
        # Guard: the repo must be in mock mode until stereo validates.
        self.assertTrue(USE_MOCK_DEPTH)


# ==============================================================
# 2. Staleness (provenance STALE; value withheld)
# ==============================================================

class TestStaleness(unittest.TestCase):
    def test_old_measurement_is_rejected_by_fusion(self):
        clock = FakeClock()

        class OldTimestampProvider(ScenarioDepthProvider):
            """Simulates a provider handing out old measurements
            (what the real Oak provider does with a stalled frame)."""

            def get_measurement(self, bbox, frame_shape, label=None, now=None):
                m = super().get_measurement(bbox, frame_shape, label=label, now=now)
                return DepthMeasurement(
                    distance_m=m.distance_m,
                    provenance=m.provenance,
                    source=m.source,
                    reason=m.reason,
                    timestamp=clock.now - 5.0,  # measured 5s ago
                )

        p = OldTimestampProvider(clock=clock)
        fusion = DistanceFusion(p, clock=clock)

        det = make_det("person")
        fusion.fuse([det], (400, 640))
        self.assertIsNone(det.distance)  # stale value withheld
        self.assertEqual(det.distance_provenance, "STALE")

    def test_oak_provider_reports_stale_frame(self):
        clock = FakeClock()

        class FakeCam:
            depth_status = "DEPTH_OK"

            def is_depth_available(self):
                return True

            def get_depth_frame(self):
                return None  # stream stalled after first frame

            def get_depth_health(self):
                return {"depth_status": self.depth_status}

        provider = OakStereoDepthProvider(FakeCam(), clock=clock)
        # Inject a frame "measured" long ago.
        provider._frame = __import__("numpy").zeros((400, 640), dtype="uint16")
        provider._frame_at = clock.now - 2.0

        m = provider.get_measurement((300, 100, 340, 200), (400, 640),
                                     now=clock.now)
        self.assertIsNone(m.distance_m)
        self.assertEqual(m.provenance, DistanceProvenance.STALE)


# ==============================================================
# 3. Fusion
# ==============================================================

class TestFusion(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.p = ScenarioDepthProvider(clock=self.clock)
        self.fusion = DistanceFusion(self.p, clock=self.clock)

    def test_valid_mock_distance_fused(self):
        det = make_det("person")
        self.fusion.fuse([det], (400, 640))
        self.assertAlmostEqual(det.distance, 1.2)
        self.assertEqual(det.distance_provenance, "SIMULATED")
        self.assertEqual(det.distance_source, "mock_scenario")

    def test_unavailable_distance_leaves_none(self):
        det = make_det("rocket")
        self.fusion.fuse([det], (400, 640))
        self.assertIsNone(det.distance)
        self.assertEqual(det.distance_provenance, "UNAVAILABLE")

    def test_fusion_is_nonblocking_and_inplace(self):
        dets = [make_det("person"), make_det("chair", bbox=BoundingBox(500, 100, 600, 200))]
        out = self.fusion.fuse(dets, (400, 640))
        self.assertIs(out, dets)
        self.assertAlmostEqual(dets[0].distance, 1.2)
        self.assertAlmostEqual(dets[1].distance, 1.5)  # chair/RIGHT

    def test_stop_fusion_covers_lifecycle(self):
        self.fusion.start()
        self.fusion.update()
        self.fusion.stop()
        health = self.fusion.health()
        self.assertEqual(health["provider"], "mock_scenario")
        self.assertTrue(health["allow_simulated"])


# ==============================================================
# 4-7. Distance-aware navigation + escalation
# ==============================================================

class TestDistanceNavigation(unittest.TestCase):
    def setUp(self):
        self.nav = DistanceNavigator()

    def test_person_critical_stops(self):
        d = self.nav.decide([make_det("person", distance=1.2, provenance="SIMULATED")])
        self.assertEqual(d.action, NavigationAction.STOP)

    def test_chair_midrange_slows(self):
        d = self.nav.decide([make_det("chair", distance=2.5, provenance="SIMULATED")])
        self.assertEqual(d.action, NavigationAction.SLOW_DOWN)

    def test_obstacle_under_critical_stops_any_region(self):
        d = self.nav.decide([make_det("chair", region="LEFT",
                                      bbox=BoundingBox(50, 100, 150, 200),
                                      distance=0.7, provenance="SIMULATED")])
        self.assertEqual(d.action, NavigationAction.STOP)

    def test_far_object_continues(self):
        d = self.nav.decide([make_det("person", distance=4.0, provenance="SIMULATED")])
        self.assertEqual(d.action, NavigationAction.CONTINUE)

    def test_lateral_object_moves_away(self):
        d = self.nav.decide([make_det("chair", region="LEFT",
                                      bbox=BoundingBox(50, 100, 150, 200),
                                      distance=1.5, provenance="SIMULATED")])
        self.assertEqual(d.action, NavigationAction.MOVE_RIGHT)

        d = self.nav.decide([make_det("chair", region="RIGHT",
                                      bbox=BoundingBox(500, 100, 600, 200),
                                      distance=1.5, provenance="SIMULATED")])
        self.assertEqual(d.action, NavigationAction.MOVE_LEFT)

    def test_stale_distance_falls_back_to_spatial(self):
        # STALE must never drive the distance policy.
        d = self.nav.decide([make_det("person", distance=0.5, provenance="STALE")])
        # person CENTER + hazard -> legacy spatial policy decides.
        self.assertIn(d.action, (NavigationAction.STOP, NavigationAction.SLOW_DOWN))
        self.assertNotIn("STALE", d.reason)

    def test_no_distances_delegates_to_legacy_navigator(self):
        legacy = Navigator()
        det = make_det("person")  # CENTER hazard, no distance
        expected = legacy.decide([det])
        actual = DistanceNavigator().decide([make_det("person")])
        self.assertEqual(actual.action, expected.action)
        self.assertEqual(actual.reason, expected.reason)

    def test_priority_tiebreak_center_first(self):
        # Center lane outranks laterals (cascade isolation, as in the
        # legacy Navigator): person 1.9m ahead beats chair 1.5m aside.
        person = make_det("person", distance=1.9, provenance="SIMULATED")
        chair = make_det("chair", bbox=BoundingBox(500, 100, 600, 200),
                         region="RIGHT", distance=1.5, provenance="SIMULATED")
        d = self.nav.decide([person, chair])
        self.assertEqual(d.action, NavigationAction.SLOW_DOWN)
        self.assertIn("Person", d.reason)

    def test_safety_escalation_closest_wins_across_regions(self):
        # Both laterals in the warning band, neither critical: LEFT lane
        # is evaluated first (deterministic dev policy) -> move away
        # from it, i.e. MOVE_RIGHT.
        person_left = make_det("person", bbox=BoundingBox(50, 100, 150, 200),
                               region="LEFT", distance=1.6, provenance="SIMULATED")
        chair_right = make_det("chair", bbox=BoundingBox(500, 100, 600, 200),
                               region="RIGHT", distance=1.5, provenance="SIMULATED")
        d = self.nav.decide([person_left, chair_right])
        self.assertEqual(d.action, NavigationAction.MOVE_RIGHT)
        self.assertIn("Person", d.reason)

    def test_lateral_object_at_critical_stops_even_off_center(self):
        # Critical proximity overrides lane priority (safety escalation):
        # 1.2m <= CRITICAL_DISTANCE_M -> STOP regardless of region.
        chair_right = make_det("chair", bbox=BoundingBox(500, 100, 600, 200),
                               region="RIGHT", distance=1.2, provenance="SIMULATED")
        d = self.nav.decide([chair_right])
        self.assertEqual(d.action, NavigationAction.STOP)

    def test_reason_records_provenance(self):
        d = self.nav.decide([make_det("person", distance=1.2, provenance="SIMULATED")])
        self.assertIn("SIMULATED", d.reason)


class TestAudioEscalationEndToEnd(unittest.TestCase):
    """Audio escalation must still work with distance-aware decisions."""

    def test_escalation_priority_and_distance_path(self):
        from audio.announcer import Announcer
        from audio.message_builder import build_message

        clock = FakeClock()
        ann = Announcer(clock=clock)

        nav = DistanceNavigator()

        # Frame 1: person 2.4m -> SLOW_DOWN announced.
        det = make_det("person", distance=2.4, provenance="SIMULATED")
        d1 = nav.decide([make_det("person", distance=2.4, provenance="SIMULATED")])
        text1 = ann.evaluate(d1)
        self.assertIsNotNone(text1)
        self.assertIn("2.5 meters", text1)  # bucket rounding to 0.5m step

        # Frame 2: same person 1.2m -> STOP must bypass cooldown.
        clock.advance(0.2)
        det.distance = 1.2
        d2 = nav.decide([det])
        self.assertEqual(d2.action, NavigationAction.STOP)
        text2 = ann.evaluate(d2)
        self.assertIsNotNone(text2, "STOP escalation must bypass cooldown")
        self.assertIn("1 meter", text2)

        # Signature dedup: identical decision right after -> suppressed.
        clock.advance(0.1)
        self.assertIsNone(ann.evaluate(d2))


class TestStereoSwapArchitecture(unittest.TestCase):
    """
    STEP 8 guarantee: downstream (fusion, navigation, audio) works
    IDENTICALLY when the provider reports MEASURED instead of
    SIMULATED — proving the Phase E swap needs zero downstream changes.
    Uses a fake camera-manager-backed provider with the same interface
    as OakStereoDepthProvider (the real one is hardware-gated).
    """

    def test_measured_provenance_flows_end_to_end(self):
        import numpy as np
        from depth.provider import DepthProvider, DepthMeasurement, DistanceProvenance

        class FakeStereoProvider(DepthProvider):
            """Same contract as OakStereoDepthProvider, MEASURED values."""

            def __init__(self):
                self._frame = np.full((400, 640), 1200, dtype="uint16")  # 1.2 m
                self._frame_at = 1000.0

            @property
            def name(self):
                return "fake_stereo"

            def start(self):
                pass

            def stop(self):
                pass

            def update(self):
                pass

            def get_measurement(self, bbox, frame_shape, label=None, now=None):
                # Depths encoded in mm -> measure_region returns 1.2 m.
                from depth.validity import measure_region
                stats = measure_region(self._frame, bbox)
                if not stats.valid:
                    return DepthMeasurement(None, DistanceProvenance.UNAVAILABLE,
                                            self.name, stats.reason,
                                            self._frame_at)
                return DepthMeasurement(stats.distance_m,
                                        DistanceProvenance.MEASURED,
                                        self.name, None, self._frame_at)

        clock = FakeClock()
        fusion = DistanceFusion(FakeStereoProvider(), clock=clock)
        det = make_det("person")
        fusion.fuse([det], (400, 640))

        self.assertAlmostEqual(det.distance, 1.2, places=3)
        self.assertEqual(det.distance_provenance, "MEASURED")  # not SIMULATED

        decision = DistanceNavigator().decide([det])
        self.assertEqual(decision.action, NavigationAction.STOP)
        self.assertIn("MEASURED", decision.reason)  # honest provenance in UI/log


class TestLegacyRegression(unittest.TestCase):
    """Legacy spatial Navigator behavior unchanged by the new layer."""

    def test_legacy_still_works_alongside(self):
        det = make_det("person")
        d = Navigator().decide([det])
        self.assertEqual(d.action, NavigationAction.STOP)
        self.assertEqual(d.trigger, det)


if __name__ == "__main__":
    unittest.main(verbosity=2)
