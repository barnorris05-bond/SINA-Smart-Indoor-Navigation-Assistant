"""
evaluation/record.py

MEASUREMENT RECORD SCHEMA (offline evaluation infrastructure).

One immutable record per (frame, object) distance observation. This is
the interchange format between:

    live SINA run (future MEASURED OAK-D data)  -> records -> evaluation
    offline replay / failure injection (SIMULATED) -> records -> evaluation

Design rules (milestone requirements):

- Ground truth is a SEPARATE field, never fabricated: a missing ground
  truth is recorded as None, never as 0 or a guessed value.
- Provenance strings reuse the project vocabulary exactly
  ("MEASURED" | "SIMULATED" | "UNAVAILABLE" | "STALE", from
  depth/provider.DistanceProvenance). Nothing here reinterprets them.
- JSONL is the primary format (append-friendly, one object per line,
  deterministic, diff-able); a CSV header/row pair is provided for
  spreadsheet inspection. No database.

SCHEMA_VERSION=1 fields:

    schema_version          int
    scene_id                str   logical capture session / scenario id
    frame_id                int   monotonically increasing frame number
    timestamp               float seconds (injectable clock in replay)
    track_id                Optional[int]   None = tracker not run
    label                   str
    confidence              float
    bbox                    (x1, y1, x2, y2) ints
    region                  "LEFT" | "CENTER" | "RIGHT"
    predicted_distance_m    Optional[float] None = no usable value
    distance_provenance     str (project vocabulary, see above)
    distance_source         Optional[str]   provider id
    ground_truth_distance_m Optional[float] None = not measured
"""

import json
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

SCHEMA_VERSION = 1

VALID_PROVENANCES = ("MEASURED", "SIMULATED", "UNAVAILABLE", "STALE")


@dataclass(frozen=True)
class MeasurementRecord:
    """One distance observation with honest provenance and optional truth."""

    schema_version: int
    scene_id: str
    frame_id: int
    timestamp: float
    track_id: Optional[int]
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    region: str
    predicted_distance_m: Optional[float]
    distance_provenance: str
    distance_source: Optional[str]
    ground_truth_distance_m: Optional[float]

    # ------------------------------------------------------
    # JSON (primary)
    # ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "track_id": self.track_id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "region": self.region,
            "predicted_distance_m": self.predicted_distance_m,
            "distance_provenance": self.distance_provenance,
            "distance_source": self.distance_source,
            "ground_truth_distance_m": self.ground_truth_distance_m,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def parse_json(cls, line: str) -> "MeasurementRecord":
        """Deserialize one JSON line (round-trip of to_json())."""
        return cls.from_dict(json.loads(line))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeasurementRecord":
        """
        Deserialize one record.

        - bbox accepts list or tuple.
        - schema_version must be <= SCHEMA_VERSION (forward records from
          a newer schema are rejected rather than silently misread).
        - missing optional fields are an error EXCEPT ground truth /
          distance_source / track_id which may be explicitly None.
        """
        if int(data["schema_version"]) > SCHEMA_VERSION:
            raise ValueError(
                f"record schema_version {data['schema_version']} newer than "
                f"supported {SCHEMA_VERSION}"
            )
        bbox = tuple(int(v) for v in data["bbox"])
        if len(bbox) != 4:
            raise ValueError(f"bbox must have 4 values, got {bbox!r}")
        provenance = str(data["distance_provenance"])
        if provenance not in VALID_PROVENANCES:
            raise ValueError(f"unknown provenance {provenance!r}")
        return cls(
            schema_version=int(data["schema_version"]),
            scene_id=str(data["scene_id"]),
            frame_id=int(data["frame_id"]),
            timestamp=float(data["timestamp"]),
            track_id=(
                None if data["track_id"] is None else int(data["track_id"])
            ),
            label=str(data["label"]),
            confidence=float(data["confidence"]),
            bbox=bbox,  # type: ignore[arg-type]
            region=str(data["region"]),
            predicted_distance_m=(
                None if data["predicted_distance_m"] is None
                else float(data["predicted_distance_m"])
            ),
            distance_provenance=provenance,
            distance_source=(
                None if data.get("distance_source") is None
                else str(data["distance_source"])
            ),
            ground_truth_distance_m=(
                None if data.get("ground_truth_distance_m") is None
                else float(data["ground_truth_distance_m"])
            ),
        )

    # ------------------------------------------------------
    # CSV (inspection convenience)
    # ------------------------------------------------------

    @staticmethod
    def csv_header() -> str:
        return ",".join(f.name for f in fields(MeasurementRecord))

    def to_csv_row(self) -> str:
        d = self.to_dict()
        d["bbox"] = ";".join(str(v) for v in self.bbox)
        return ",".join(
            "" if d[f.name] is None else str(d[f.name])
            for f in fields(MeasurementRecord)
        )

    # ------------------------------------------------------
    # Identity (calibration / validation overlap checks)
    # ------------------------------------------------------

    def identity_key(self) -> Tuple[str, int, Optional[int]]:
        """(scene_id, frame_id, track_id) - one observation's identity."""
        return (self.scene_id, self.frame_id, self.track_id)
