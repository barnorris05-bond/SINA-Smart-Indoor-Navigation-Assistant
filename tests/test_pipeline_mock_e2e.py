"""
tests/test_pipeline_mock_e2e.py

END-TO-END pipeline test WITHOUT any hardware (Phase B-D validation).

Chain under test (real classes; only the detector's *pixels* are
synthetic, exactly what hardware otherwise provides):

    FakeDetector (stands in for YOLODetector on synthetic frames)
        -> DetectionManager.process()          (enrichment, regions)
        -> DistanceFusion.fuse()               (MOCK distances, stamped)
        -> DistanceNavigator.decide()          (distance-aware policy)
        -> decision log + audio.evaluate()     (gated announcements)

No OAK-D, no DepthAI, no OpenCV windows, no pyttsx3 speaker thread.

Run:
    ./.venv/Scripts/python.exe tests/test_pipeline_mock_e2e.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.object_detector import ObjectDetector, DetectedObject, BoundingBox
from vision.detection_manager import DetectionManager
from vision.distance_fusion import DistanceFusion
from depth.mock_provider import ScenarioDepthProvider
from navigation.distance_navigator import DistanceNavigator
from navigation.navigation_types import NavigationAction
from audio.announcer import Announcer


# ==============================================================
# Synthetic frame + detector (hardware stand-in)
# ==============================================================

FRAME_W, FRAME_H = 640, 400  # production RGB geometry


def person_frame(center_x=320, box_w=60, box_h=200, bright=220):
    """Dark scene, bright rectangle standing in for a person."""
    import numpy as np
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    x1 = center_x - box_w // 2
    y1 = 150
    x2 = x1 + box_w
    y2 = y1 + box_h
    frame[y1:y2, x1:x2] = bright
    return frame


class FakeDetector(ObjectDetector):
    """Returns the ground-truth object the frame was built with."""

    def load_model(self):
        pass

    def detect(self, frame):
        return [DetectedObject(
            label="person",
            confidence=0.93,
            bbox=BoundingBox(FRAME_W // 2 - 30, 150,
                             FRAME_W // 2 + 30, 350),
        )]


# ==============================================================
# End-to-end scenarios
# ==============================================================

class TestPipelineMockE2E(unittest.TestCase):
    def setUp(self):
        self.manager = DetectionManager()
        self.fusion = DistanceFusion(ScenarioDepthProvider())
        self.navigator = DistanceNavigator()
        self.announcer = Announcer(clock=lambda: self.t)
        self.t = 100.0
        self.detector = FakeDetector()

    def _tick(self):
        """One full pipeline pass; returns (detections, decision, text)."""
        frame = person_frame()
        raw = self.detector.detect(frame)
        dets = self.manager.process(raw, frame.shape[1])
        self.fusion.fuse(dets, frame.shape[:2])
        decision = self.navigator.decide(dets)
        text = self.announcer.evaluate(decision)
        self.t += 0.1
        return dets, decision, text

    def test_full_chain_person_center(self):
        dets, decision, _ = self._tick()
        self.assertEqual(len(dets), 1)
        det = dets[0]
        self.assertEqual(det.label, "person")
        self.assertEqual(det.region, "CENTER")
        self.assertEqual(det.priority, 100)
        self.assertAlmostEqual(det.distance, 1.2)  # mock table, SIMULATED
        self.assertEqual(det.distance_provenance, "SIMULATED")
        self.assertEqual(decision.action, NavigationAction.STOP)
        self.assertIn("SIMULATED", decision.reason)

    def test_five_frame_stream_stays_stable_and_gated(self):
        decisions = []
        texts = []
        for _ in range(5):
            dets, decision, text = self._tick()
            decisions.append(decision.action)
            texts.append(text)
        self.assertTrue(all(a == NavigationAction.STOP for a in decisions))
        # Gating: 5 identical frames -> 1 announcement (dedup cooldown),
        # escalations would still pass.
        self.assertEqual(texts.count(None), 4)
        self.assertEqual(len([t for t in texts if t is not None]), 1)

    def test_distance_unavailable_falls_back_to_spatial(self):
        # A label the scenario table doesn't know: no distance flows.
        raw = [DetectedObject(label="rocket", confidence=0.9,
                              bbox=BoundingBox(300, 150, 340, 350))]
        dets = self.manager.process(raw, FRAME_W)
        self.fusion.fuse(dets, (FRAME_H, FRAME_W))
        self.assertEqual(dets[0].distance_provenance, "UNAVAILABLE")
        self.assertIsNone(dets[0].distance)
        decision = self.navigator.decide(dets)
        # Legacy spatial fallback: non-hazard object in CENTER ->
        # SLOW_DOWN (identical to the pre-mock-depth Navigator).
        self.assertEqual(decision.action, NavigationAction.SLOW_DOWN)
        self.assertIn("rocket", decision.reason.lower())

    def test_audio_text_carries_mock_distance(self):
        _, decision, text = self._tick()
        self.assertIsNotNone(text)
        self.assertIn("1 meter", text)  # 1.2m rounds to the 0.5m bucket "1"

    def test_recorder_captures_provenance_for_evaluation(self):
        # Minimal in-test recorder showing what Phase F evaluation would
        # log per frame (no files written).
        rows = []
        for _ in range(3):
            dets, decision, _ = self._tick()
            rows.append({
                "action": decision.action.name,
                "reason": decision.reason,
                "label": dets[0].label if dets else None,
                "distance": dets[0].distance if dets else None,
                "provenance": dets[0].distance_provenance if dets else None,
            })
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["provenance"] == "SIMULATED" for r in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
