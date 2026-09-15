"""
evaluation/recorder.py

LIVE CAPTURE HELPER.

Converts the annotated pipeline output of a running SINA instance
(detections + decision per frame) into MeasurementRecords plus an
EventLogger trace. This is the future MEASURED-data path: when real
OAK-D stereo is validated, a capture session feeds these objects from
main.py and the resulting JSONL goes straight into evaluate().

Ground truth: attach_truth() joins ground-truth distances by
(scene_id, frame_id, track_id) — the manual-measurement workflow of
Phase 10. Ground truth is NEVER invented here: records leave the
recorder with ground_truth_distance_m=None.
"""

from typing import Dict, Iterable, List, Optional, Tuple

from vision.object_detector import DetectedObject
from navigation.navigator import NavigationDecision

from evaluation.record import MeasurementRecord, SCHEMA_VERSION
from evaluation.event_log import EventLogger

TruthKey = Tuple[str, int, Optional[int]]


class MeasurementRecorder:
    """Accumulates MeasurementRecords + events from live/replayed frames."""

    def __init__(
        self,
        scene_id: str,
        event_sink=None,
    ) -> None:
        self.scene_id = scene_id
        self.events = EventLogger(sink=event_sink)
        self.records: List[MeasurementRecord] = []

    # ------------------------------------------------------
    # Capture
    # ------------------------------------------------------

    def capture_frame(
        self,
        frame_id: int,
        timestamp: float,
        detections: List[DetectedObject],
        decision: NavigationDecision,
    ) -> List[MeasurementRecord]:
        """
        Record one frame: structured event + one record per detection.
        Provenance is copied verbatim from the pipeline stamping.
        """
        self.events.log_frame(frame_id, timestamp, detections, decision)
        out: List[MeasurementRecord] = []
        for det in detections:
            out.append(MeasurementRecord(
                schema_version=SCHEMA_VERSION,
                scene_id=self.scene_id,
                frame_id=frame_id,
                timestamp=timestamp,
                track_id=det.track_id,
                label=det.label,
                confidence=det.confidence,
                bbox=(det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2),
                region=det.region,
                predicted_distance_m=det.distance,
                distance_provenance=det.distance_provenance,
                distance_source=det.distance_source,
                ground_truth_distance_m=None,
            ))
        self.records.extend(out)
        return out

    # ------------------------------------------------------
    # Ground truth joining (Phase 10 workflow)
    # ------------------------------------------------------

    def attach_truth(
        self,
        truth: Dict[TruthKey, float],
    ) -> int:
        """
        Join ground-truth distances onto records by identity key.
        Returns the number of records that received a truth value.

        Records without a matching entry keep ground_truth=None
        (explicitly missing, never fabricated).
        """
        joined = 0
        for i, rec in enumerate(self.records):
            key = rec.identity_key()
            if key in truth and rec.ground_truth_distance_m is None:
                self.records[i] = MeasurementRecord(
                    schema_version=rec.schema_version,
                    scene_id=rec.scene_id,
                    frame_id=rec.frame_id,
                    timestamp=rec.timestamp,
                    track_id=rec.track_id,
                    label=rec.label,
                    confidence=rec.confidence,
                    bbox=rec.bbox,
                    region=rec.region,
                    predicted_distance_m=rec.predicted_distance_m,
                    distance_provenance=rec.distance_provenance,
                    distance_source=rec.distance_source,
                    ground_truth_distance_m=float(truth[key]),
                )
                joined += 1
        return joined

    def close(self) -> None:
        self.events.close()
