"""
calibration/session.py

EXPERIMENT SESSION METADATA (Phase 10 preparation).

A lightweight, serializable record of ONE controlled measurement
session: where/when/how the data was captured, on which device, at
which software version. It exists so future calibration datasets can
answer "under what conditions was this measured?" without inventing
metadata after the fact.

Rules (milestone §5):

- Only fields that can be RELIABLY collected. Every field is optional:
  absent conditions are recorded as None, never guessed or defaulted.
- Deterministic serialization (sorted keys) for diff-able files.
- No safety semantics. A session is bookkeeping, not a claim.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ExperimentSession:
    """
    Metadata for one controlled measurement session.

    All fields optional: an unknown value is None, never invented.
    Notes are free-text operator observations (lighting, surfaces,
    procedure deviations); they are descriptive, not parsed.
    """

    session_id: Optional[str] = None
    date_time: Optional[str] = None          # ISO-8601 string or None
    device_id: Optional[str] = None          # e.g. OAK-D serial
    device_model: Optional[str] = None       # e.g. OAK-D Lite (RVC2)
    software_version: Optional[str] = None   # git commit / tag
    depthai_version: Optional[str] = None
    rgb_resolution: Optional[str] = None     # e.g. "640x400"
    stereo_resolution: Optional[str] = None  # e.g. "640x480"
    fps: Optional[int] = None
    scene_id: Optional[str] = None           # links to MeasurementRecord.scene_id
    lighting: Optional[str] = None           # free-text description
    surfaces: Optional[str] = None           # free-text description
    target_distance_m: Optional[float] = None  # nominal bench distance
    operator_notes: Optional[str] = None
    tags: Tuple[str, ...] = field(default_factory=tuple)  # free-form labels

    # ------------------------------------------------------
    # Serialization
    # ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "session_id": self.session_id,
            "date_time": self.date_time,
            "device_id": self.device_id,
            "device_model": self.device_model,
            "software_version": self.software_version,
            "depthai_version": self.depthai_version,
            "rgb_resolution": self.rgb_resolution,
            "stereo_resolution": self.stereo_resolution,
            "fps": self.fps,
            "scene_id": self.scene_id,
            "lighting": self.lighting,
            "surfaces": self.surfaces,
            "target_distance_m": self.target_distance_m,
            "operator_notes": self.operator_notes,
            "tags": list(self.tags),
        }
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentSession":
        tags = data.get("tags") or []
        if not isinstance(tags, (list, tuple)) or not all(
            isinstance(t, str) for t in tags
        ):
            raise ValueError(f"tags must be a list of strings, got {tags!r}")
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown session fields: {sorted(unknown)}")
        return cls(
            session_id=_opt_str(data, "session_id"),
            date_time=_opt_str(data, "date_time"),
            device_id=_opt_str(data, "device_id"),
            device_model=_opt_str(data, "device_model"),
            software_version=_opt_str(data, "software_version"),
            depthai_version=_opt_str(data, "depthai_version"),
            rgb_resolution=_opt_str(data, "rgb_resolution"),
            stereo_resolution=_opt_str(data, "stereo_resolution"),
            fps=None if data.get("fps") is None else int(data["fps"]),
            scene_id=_opt_str(data, "scene_id"),
            lighting=_opt_str(data, "lighting"),
            surfaces=_opt_str(data, "surfaces"),
            target_distance_m=(
                None if data.get("target_distance_m") is None
                else float(data["target_distance_m"])
            ),
            operator_notes=_opt_str(data, "operator_notes"),
            tags=tuple(str(t) for t in tags),
        )

    @classmethod
    def parse_json(cls, line: str) -> "ExperimentSession":
        """Round-trip of to_json()."""
        return cls.from_dict(json.loads(line))


def _opt_str(data: Dict[str, Any], key: str) -> Optional[str]:
    value = data.get(key)
    return None if value is None else str(value)
