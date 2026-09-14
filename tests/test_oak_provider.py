"""
tests/test_oak_provider.py

PHASE 9 — REAL PROVIDER UNIT TESTS (hardware-free).

Exercises OakStereoDepthProvider's PURE LOGIC with synthetic depth
arrays and a fake CameraManager — no OAK-D, no DepthAI import, no
device. (depth/oak_provider.py deliberately imports no DepthAI.)

Covers (master roadmap §14-16, §21):

  §15  valid ROI -> MEASURED distance
       zero/invalid pixels rejected
       outliers do not dominate (median)
       empty ROI -> UNAVAILABLE
       insufficient valid pixels -> UNAVAILABLE
       provenance correctness
       coordinate mapping (equal and differing frame sizes)
       no stale promotion
       multiple objects measured independently
  §16  RGB/depth alignment on synthetic depth maps:
       2.0 m region vs 5.0 m region selected correctly by RGB bbox;
       bbox at left/right/top/bottom edges, partially outside, tiny box
  §21  failure paths: stream down, no frame yet, stale frame age,
       multiple objects, stop() clears cached frame

Run:
    ./.venv/Scripts/python.exe tests/test_oak_provider.py
"""

import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth.provider import DistanceProvenance
from depth.oak_provider import OakStereoDepthProvider

RGB_H, RGB_W = 400, 640


class FakeCameraManager:
    """
    Hardware-free stand-in exposing exactly the surface
    OakStereoDepthProvider consumes (duck-typed, no DepthAI).
    """

    def __init__(self, depth_available=True, frame=None):
        self.depth_status = "DEPTH_OK" if depth_available else "DEPTH_DISABLED"
        self._available = depth_available
        self._frame = frame
        self.update_calls = 0

    def is_depth_available(self) -> bool:
        return self._available

    def get_depth_frame(self):
        self.update_calls += 1
        return self._frame

    def get_depth_health(self) -> dict:
        return {"depth_status": self.depth_status, "frame_count": 1}

    # OakStereoDepthProvider reads depth_status for the UNAVAILABLE reason.
    @property
    def depth_status(self):
        return self._status

    @depth_status.setter
    def depth_status(self, value):
        self._status = value


def depth_frame(shape=(RGB_H, RGB_W), mm=2000, invalid_box=None):
    """Synthetic aligned depth frame (uint16 mm) with optional invalid patch."""
    frame = np.full(shape, mm, dtype=np.uint16)
    if invalid_box is not None:
        x1, y1, x2, y2 = invalid_box
        frame[y1:y2, x1:x2] = 0
    return frame


class ValidMeasurementTest(unittest.TestCase):
    """§15.1-6: MEASURED distances from synthetic frames."""

    def setUp(self):
        self.cam = FakeCameraManager(frame=depth_frame(mm=2000))
        self.provider = OakStereoDepthProvider(self.cam, clock=lambda: 1000.0)
        self.provider.update()
        self.bbox = (280, 140, 360, 260)  # central box of a 640x400 frame

    def test_valid_roi_returns_measured_distance(self):
        m = self.provider.get_measurement(self.bbox, (RGB_H, RGB_W))
        self.assertEqual(m.provenance, DistanceProvenance.MEASURED)
        self.assertEqual(m.source, "oak_stereo")
        self.assertAlmostEqual(m.distance_m, 2.0, places=3)

    def test_zero_pixels_rejected(self):
        # Invalid patch exactly over the bbox ROI.
        self.cam._frame = depth_frame(mm=2000, invalid_box=(250, 120, 390, 280))
        self.provider.update()
        m = self.provider.get_measurement(self.bbox, (RGB_H, RGB_W))
        self.assertEqual(m.provenance, DistanceProvenance.UNAVAILABLE)
        self.assertIsNone(m.distance_m)
        self.assertIn("region", m.reason)

    def test_outliers_do_not_dominate_median(self):
        # Minority outliers (< 50% of the central ROI) are rejected by
        # the median; a near patch covering MOST of the ROI would
        # legitimately dominate (the object is nearer). Patch here is
        # ~33% of the central ROI (300,170,340,230).
        frame = np.full((RGB_H, RGB_W), 2000, dtype=np.uint16)
        frame[160:190, 300:340] = 300
        self.cam._frame = frame
        self.provider.update()
        m = self.provider.get_measurement(self.bbox, (RGB_H, RGB_W))
        self.assertEqual(m.provenance, DistanceProvenance.MEASURED)
        self.assertAlmostEqual(m.distance_m, 2.0, places=3)

    def test_empty_roi_unavailable(self):
        # Fully off-frame box cannot be measured.
        m = self.provider.get_measurement((-100, -100, -50, -50), (RGB_H, RGB_W))
        self.assertEqual(m.provenance, DistanceProvenance.UNAVAILABLE)
        self.assertIsNone(m.distance_m)

    def test_insufficient_valid_pixels_unavailable(self):
        # Everything invalid inside the ROI -> UNAVAILABLE, not 0 m.
        self.cam._frame = depth_frame(mm=2000, invalid_box=(200, 100, 440, 300))
        self.provider.update()
        m = self.provider.get_measurement(self.bbox, (RGB_H, RGB_W))
        self.assertEqual(m.provenance, DistanceProvenance.UNAVAILABLE)

    def test_out_of_range_values_treated_invalid(self):
        # 25 m readings are beyond DEPTH_MAX_MM -> invalid; whole ROI
        # invalid -> UNAVAILABLE rather than a bogus distance.
        frame = np.full((RGB_H, RGB_W), 25000, dtype=np.uint16)
        self.cam._frame = frame
        self.provider.update()
        m = self.provider.get_measurement(self.bbox, (RGB_H, RGB_W))
        self.assertEqual(m.provenance, DistanceProvenance.UNAVAILABLE)

    def test_mixed_frame_median_uses_valid_pixels(self):
        # Half the frame invalid: bbox in the valid half still measures.
        frame = depth_frame(mm=1500)
        frame[:RGB_H // 2, :] = 0
        self.cam._frame = frame
        self.provider.update()
        m = self.provider.get_measurement((280, 250, 360, 350), (RGB_H, RGB_W))
        self.assertEqual(m.provenance, DistanceProvenance.MEASURED)
        self.assertAlmostEqual(m.distance_m, 1.5, places=3)


class AlignmentMappingTest(unittest.TestCase):
    """
    §16: RGB bbox -> depth region mapping on synthetic scenes.

    Production geometry note (documented, per master §8): the stereo
    branch requests monos at 640x480 while RGB is 640x400, and depth is
    aligned to CAM_A. DepthAI crops the vertical FOV to match the RGB
    aspect when aligning, so the aligned depth frame is expected at
    640x400; the provider additionally SCALES defensively whenever the
    depth frame size differs from the RGB frame, which these tests
    prove with synthetic arrays.
    """

    def setUp(self):
        # Depth deliberately at a DIFFERENT size than RGB to prove the
        # defensive scaling branch in get_measurement().
        self.cam = FakeCameraManager(
            frame=depth_frame(shape=(480, 640), mm=2000)
        )
        self.provider = OakStereoDepthProvider(self.cam, clock=lambda: 1000.0)
        self.provider.update()

    def _m(self, bbox):
        return self.provider.get_measurement(bbox, (RGB_H, RGB_W))

    def test_two_region_scene_selects_correct_region(self):
        # §16 canonical case: left half 2.0 m, right half 5.0 m.
        frame = np.full((480, 640), 5000, dtype=np.uint16)
        frame[:, :320] = 2000
        self.cam._frame = frame
        self.provider.update()
        left_bbox = (60, 100, 280, 300)      # RGB coords, left region
        right_bbox = (360, 100, 580, 300)    # RGB coords, right region
        self.assertAlmostEqual(self._m(left_bbox).distance_m, 2.0, places=3)
        self.assertAlmostEqual(self._m(right_bbox).distance_m, 5.0, places=3)

    def test_bbox_at_left_edge(self):
        m = self._m((0, 150, 120, 250))
        self.assertEqual(m.provenance, DistanceProvenance.MEASURED)
        self.assertAlmostEqual(m.distance_m, 2.0, places=3)

    def test_bbox_at_right_edge(self):
        m = self._m((520, 150, 639, 250))
        self.assertEqual(m.provenance, DistanceProvenance.MEASURED)

    def test_bbox_at_top_edge(self):
        m = self._m((270, 0, 370, 60))
        self.assertEqual(m.provenance, DistanceProvenance.MEASURED)

    def test_bbox_at_bottom_edge(self):
        m = self._m((270, 340, 370, 399))
        self.assertEqual(m.provenance, DistanceProvenance.MEASURED)

    def test_partially_outside_bbox_clamped(self):
        m = self._m((-40, 150, 100, 250))
        self.assertEqual(m.provenance, DistanceProvenance.MEASURED)

    def test_tiny_bbox_rejected_below_min_samples(self):
        # A 4x4 px RGB box scales to a ~4 px central ROI — far below
        # DEPTH_MIN_SAMPLES (30): honest UNAVAILABLE, never a junk
        # distance from 4 pixels (Phase 2 validity gate).
        m = self._m((318, 198, 322, 202))
        self.assertEqual(m.provenance, DistanceProvenance.UNAVAILABLE)
        self.assertIn("too_few_samples", m.reason)

    def test_small_but_sufficient_bbox_measures(self):
        # 16x16 px box -> ~10x12 scaled central ROI ≈ 120 px ≥ 30:
        # small boxes above the validity gate measure fine.
        m = self._m((312, 192, 328, 208))
        self.assertEqual(m.provenance, DistanceProvenance.MEASURED)
        self.assertAlmostEqual(m.distance_m, 2.0, places=3)

    def test_uniform_frame_any_bbox_same_distance(self):
        for bbox in [(0, 0, 640, 400), (270, 150, 370, 250), (600, 0, 640, 400)]:
            self.assertAlmostEqual(self._m(bbox).distance_m, 2.0, places=3)


class FailurePathTest(unittest.TestCase):
    """§21: stream down, no frame, stale age, lifecycle honesty."""

    def setUp(self):
        self.now = 1000.0

    def _provider(self, cam, clock=None):
        p = OakStereoDepthProvider(cam, clock=clock or (lambda: self.now))
        p.update()
        return p

    def test_stream_down_unavailable(self):
        cam = FakeCameraManager(depth_available=False)
        p = self._provider(cam)
        m = p.get_measurement((280, 140, 360, 260), (RGB_H, RGB_W))
        self.assertEqual(m.provenance, DistanceProvenance.UNAVAILABLE)
        self.assertIn("depth_stream", m.reason)

    def test_no_frame_yet_unavailable(self):
        cam = FakeCameraManager(depth_available=True, frame=None)
        p = self._provider(cam)
        m = p.get_measurement((280, 140, 360, 260), (RGB_H, RGB_W))
        self.assertEqual(m.provenance, DistanceProvenance.UNAVAILABLE)
        self.assertIn("no_frame_yet", m.reason)

    def test_stale_frame_never_promoted(self):
        # Frame arrived at t=1000; query at t=1002 (> 0.5 s window):
        # must be STALE, and distance_m must be None (no silent reuse).
        cam = FakeCameraManager(frame=depth_frame(mm=2000))
        p = OakStereoDepthProvider(cam, clock=lambda: 1000.0)
        p.update()
        m = p.get_measurement((280, 140, 360, 260), (RGB_H, RGB_W),
                              now=1002.0)
        self.assertEqual(m.provenance, DistanceProvenance.STALE)
        self.assertIsNone(m.distance_m)

    def test_stale_then_refresh_returns_measured(self):
        cam = FakeCameraManager(frame=depth_frame(mm=2000))
        p = OakStereoDepthProvider(cam, clock=lambda: 1000.0)
        p.update()
        m_old = p.get_measurement((280, 140, 360, 260), (RGB_H, RGB_W),
                                  now=1002.0)
        self.assertEqual(m_old.provenance, DistanceProvenance.STALE)
        self.cam_new_frame = depth_frame(mm=1500)
        cam._frame = self.cam_new_frame
        p.update()  # fresh frame at clock() == 1000.0 again
        m_new = p.get_measurement((280, 140, 360, 260), (RGB_H, RGB_W),
                                  now=1000.4)
        self.assertEqual(m_new.provenance, DistanceProvenance.MEASURED)
        self.assertAlmostEqual(m_new.distance_m, 1.5, places=3)

    def test_stop_clears_cached_frame(self):
        cam = FakeCameraManager(frame=depth_frame(mm=2000))
        p = self._provider(cam)
        p.stop()
        m = p.get_measurement((280, 140, 360, 260), (RGB_H, RGB_W))
        self.assertEqual(m.provenance, DistanceProvenance.UNAVAILABLE)
        self.assertIn("no_frame_yet", m.reason)

    def test_multiple_objects_measured_independently(self):
        # Two regions at different depths; two bboxes -> two answers.
        frame = np.full((RGB_H, RGB_W), 2000, dtype=np.uint16)
        frame[0:200, :] = 4000
        cam = FakeCameraManager(frame=frame)
        p = self._provider(cam)
        m_near = p.get_measurement((280, 250, 360, 350), (RGB_H, RGB_W))
        m_far = p.get_measurement((280, 20, 360, 120), (RGB_H, RGB_W))
        self.assertEqual(m_near.provenance, DistanceProvenance.MEASURED)
        self.assertEqual(m_far.provenance, DistanceProvenance.MEASURED)
        self.assertAlmostEqual(m_near.distance_m, 2.0, places=3)
        self.assertAlmostEqual(m_far.distance_m, 4.0, places=3)

    def test_health_passthrough(self):
        cam = FakeCameraManager(frame=depth_frame(mm=2000))
        p = self._provider(cam)
        h = p.health()
        self.assertEqual(h["provider"], "oak_stereo")
        self.assertEqual(h["depth_status"], "DEPTH_OK")


class IntegrationChainTest(unittest.TestCase):
    """
    §17-19: prove MEASURED data flows through the existing layers
    without changing their semantics: DistanceFusion ->
    DetectedObject -> MotionEstimator -> RiskEstimator ->
    DistanceNavigator/TemporalNavigator.
    """

    def setUp(self):
        from vision.object_detector import DetectedObject, BoundingBox
        from vision.distance_fusion import DistanceFusion
        from vision.motion_estimator import MotionEstimator
        from navigation.risk_estimator import RiskEstimator
        from navigation.distance_navigator import DistanceNavigator
        from navigation.temporal_navigator import TemporalNavigator
        self.DetectedObject = DetectedObject
        self.BoundingBox = BoundingBox
        self.DistanceFusion = DistanceFusion
        self.MotionEstimator = MotionEstimator
        self.RiskEstimator = RiskEstimator
        self.DistanceNavigator = DistanceNavigator
        self.TemporalNavigator = TemporalNavigator

    def _person(self, distance=None, motion_state=None, closing_rate=None,
                track_id=1):
        det = self.DetectedObject(
            label="person", confidence=0.93,
            bbox=self.BoundingBox(300, 150, 340, 350),
            track_id=track_id, priority=100, region="CENTER",
        )
        if motion_state is not None:
            det.motion_state = motion_state
            det.closing_rate_mps = closing_rate
        return det

    def test_measured_distance_flows_through_fusion_and_layers(self):
        t = [3000.0]  # mutable clock shared by provider/fusion/motion
        clock = lambda: t[0]
        cam = FakeCameraManager(frame=depth_frame(mm=1800))  # OAK: 1.8 m
        provider = OakStereoDepthProvider(cam, clock=clock)
        provider.update()
        fusion = self.DistanceFusion(provider, clock=clock)
        motion = self.MotionEstimator(clock=clock)
        risk = self.RiskEstimator()
        tnav = self.TemporalNavigator(self.DistanceNavigator())

        # Frame 1: measured 1.8 m.
        dets = [self._person()]
        fusion.fuse(dets, (RGB_H, RGB_W))
        det = dets[0]
        self.assertEqual(det.distance_provenance, "MEASURED")
        self.assertEqual(det.distance_source, "oak_stereo")
        self.assertAlmostEqual(det.distance, 1.8, places=3)
        motion.update(dets)   # 1 sample: honestly UNKNOWN
        risk.update(dets)
        self.assertEqual(det.risk_level, "HIGH")  # warning band + priority
        self.assertEqual(tnav.decide(dets).action.name, "SLOW_DOWN")

        # Frame 2: measured 1.0 m (cam._frame swapped, provider refresh).
        t[0] += 0.25
        cam._frame = depth_frame(mm=1000)
        provider.update()
        dets = [self._person()]
        fusion.fuse(dets, (RGB_H, RGB_W))
        self.assertEqual(dets[0].distance_provenance, "MEASURED")
        self.assertAlmostEqual(dets[0].distance, 1.0, places=3)
        motion.update(dets)
        risk.update(dets)
        decision = tnav.decide(dets)
        self.assertEqual(decision.action.name, "STOP")  # <= CRITICAL band
        self.assertEqual(decision.action.name, "STOP")  # (explicit gate step)

    def test_simulated_sample_never_promotes_motion_provenance(self):
        # §7: a SIMULATED history sample must poison the estimate
        # provenance forever: mixed history can never claim MEASURED,
        # even when the latest samples are genuinely measured.
        t = [3000.0]
        clock = lambda: t[0]
        motion = self.MotionEstimator(clock=clock)

        # Sample 1: SIMULATED (mock provider path).
        sim = self._person()
        sim.distance, sim.distance_provenance = 2.0, "SIMULATED"
        sim.distance_source = "mock_scenario"
        motion.update([sim])

        # Samples 2-4: genuinely MEASURED, approaching.
        cam = FakeCameraManager(frame=depth_frame(mm=1500))
        provider = OakStereoDepthProvider(cam, clock=clock)
        fusion = self.DistanceFusion(provider, clock=clock)
        for dt, mm in ((0.25, 1500), (0.50, 1200), (0.75, 900)):
            t[0] += 0.25
            cam._frame = depth_frame(mm=mm)
            provider.update()
            measured = self._person()
            fusion.fuse([measured], (RGB_H, RGB_W))
            self.assertEqual(measured.distance_provenance, "MEASURED")
            motion.update([measured])

        # An estimate exists but MUST NOT claim MEASURED.
        self.assertNotEqual(measured.motion_provenance, "MEASURED")
        self.assertEqual(measured.motion_provenance, "SIMULATED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
