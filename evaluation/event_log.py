"""
evaluation/event_log.py

STRUCTURED EVENT LOGGING (lightweight JSONL).

One JSON line per frame capturing what the full annotated pipeline
produced, for offline inspection alongside MeasurementRecords:

    ts, frame, action, action_reason, priority,
    objects: [{track_id, label, region, confidence, distance,
               distance_provenance, motion_state, closing_rate_mps,
               risk_level, risk_ttc_s}]

This does NOT replace utils.logger (human-readable application log);
it is a machine-readable per-frame trace for evaluation. Sink is
injectable: a file path, a writable object, or None (in-memory buffer
for tests). json.dumps(default=str) keeps it crash-free.
"""

import json
from pathlib import Path
from typing import IO, List, Optional, Union

from vision.object_detector import DetectedObject
from navigation.navigator import NavigationDecision


class EventLogger:
    """Append-only JSONL frame trace with an injectable sink."""

    def __init__(self, sink: Optional[Union[str, Path, IO]] = None) -> None:
        self._sink = sink
        self._file: Optional[IO] = None
        self._lines: List[str] = []

    # ------------------------------------------------------
    # Logging
    # ------------------------------------------------------

    def log_frame(
        self,
        frame_id: int,
        timestamp: float,
        detections: List[DetectedObject],
        decision: NavigationDecision,
    ) -> None:
        event = {
            "type": "frame",
            "ts": timestamp,
            "frame": frame_id,
            "action": decision.action.name,
            "action_reason": decision.reason,
            "priority": decision.priority,
            "objects": [self._object_entry(d) for d in detections],
        }
        self._write(json.dumps(event, sort_keys=True, default=str))

    # ------------------------------------------------------
    # Access / lifecycle
    # ------------------------------------------------------

    @property
    def lines(self) -> List[str]:
        """All written JSONL lines (in-memory sink or buffered file)."""
        return list(self._lines)

    def parsed(self) -> List[dict]:
        """All written events parsed back into dicts (tests)."""
        return [json.loads(line) for line in self._lines]

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    # ------------------------------------------------------
    # Internals
    # ------------------------------------------------------

    @staticmethod
    def _object_entry(det: DetectedObject) -> dict:
        return {
            "track_id": det.track_id,
            "label": det.label,
            "region": det.region,
            "confidence": det.confidence,
            "distance": det.distance,
            "distance_provenance": det.distance_provenance,
            "motion_state": det.motion_state,
            "closing_rate_mps": det.closing_rate_mps,
            "risk_level": det.risk_level,
            "risk_ttc_s": det.risk_ttc_s,
        }

    def _write(self, line: str) -> None:
        self._lines.append(line)
        if isinstance(self._sink, (str, Path)):
            if self._file is None:
                p = Path(self._sink)
                p.parent.mkdir(parents=True, exist_ok=True)
                self._file = p.open("a", encoding="utf-8")
            self._file.write(line + "\n")
            self._file.flush()
        elif self._sink is not None:
            self._sink.write(line + "\n")
