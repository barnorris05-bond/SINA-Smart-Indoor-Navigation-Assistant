"""
evaluation/injection.py

FAILURE INJECTION (software-only).

ScriptedDepthProvider is a DepthProvider double driven by a per-frame
script: a list of {label: spec} dicts, one entry per frame. It lets
tests and replays exercise every downstream layer (fusion, motion,
risk, temporal navigation) against deterministic failure modes WITHOUT
any hardware:

    spec                    produced DepthMeasurement
    --------------------    -------------------------------------------
    None                    UNAVAILABLE (no distance this frame)
    float v                 SIMULATED distance v
    ("MEASURED", v)         MEASURED distance v   (emulates the real
                            provider's contract so downstream
                            provenance handling can be tested; the
                            value itself is still synthetic - such
                            data must never be presented as a
                            hardware-validated result)
    ("STALE", v)            STALE: stream-quiet form (distance_m=None,
                            provenance=STALE) - the OakStereoDepthProvider
                            contract when the depth stream goes quiet
    ("OLD", v)              SIMULATED v with a timestamp older than the
                            fusion staleness window - exercises fusion's
                            defense-in-depth demotion to STALE

The script frame advances in update() (called once per vision loop by
DistanceFusion), so script frame N is served during loop frame N.
Before the first update() the FIRST script frame applies (clamped),
and beyond the last scripted frame the LAST script is held
(deterministic; document when relying on it). A label missing from a
frame's script yields UNAVAILABLE for that object.

Honesty rule: this module never mutates downstream state directly; it
only produces honest DepthMeasurement values per the table above, and
all downstream layers keep enforcing their own provenance rules.
"""

from typing import Dict, List, Optional, Tuple, Union

from depth.provider import (
    DepthProvider,
    DepthMeasurement,
    DistanceProvenance,
)
from depth.provider import BBox, FrameShape

ScriptSpec = Union[float, Tuple[str, float], None]


class ScriptedDepthProvider(DepthProvider):
    """Deterministic per-frame scripted provider (see module docstring)."""

    def __init__(
        self,
        frame_scripts: List[Dict[str, ScriptSpec]],
        name: str = "scripted_eval",
        stale_age_s: float = 5.0,
    ) -> None:
        if not frame_scripts:
            raise ValueError("frame_scripts must contain at least one frame")
        for i, script in enumerate(frame_scripts):
            if not isinstance(script, dict):
                raise TypeError(f"frame_scripts[{i}] must be a dict")
        self._scripts = frame_scripts
        self._name = name
        self._stale_age_s = stale_age_s
        self._frame_index = -1  # update() advances to 0 before frame 0

    # ------------------------------------------------------
    # DepthProvider contract
    # ------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def update(self) -> None:
        """Advance the script one frame (DistanceFusion calls this)."""
        self._frame_index += 1

    def get_measurement(
        self,
        bbox: BBox,
        frame_shape: FrameShape,
        label: Optional[str] = None,
        now: Optional[float] = None,
    ) -> DepthMeasurement:
        now = 0.0 if now is None else now
        # max(0, ...) guards pre-first-update(): Python's negative
        # indexing must not reach into the script from the end.
        script = self._scripts[max(0, min(self._frame_index, len(self._scripts) - 1))]
        spec = script.get(label) if label is not None else None
        return self._materialize(spec, now)

    def health(self) -> dict:
        return {
            "provider": self._name,
            "ok": True,
            "frame_index": self._frame_index,
            "scripted_frames": len(self._scripts),
        }

    # ------------------------------------------------------
    # Internals
    # ------------------------------------------------------

    def _materialize(self, spec: ScriptSpec, now: float) -> DepthMeasurement:
        if spec is None:
            return DepthMeasurement(
                distance_m=None,
                provenance=DistanceProvenance.UNAVAILABLE,
                source=self._name,
                reason="scripted_unavailable",
                timestamp=now,
            )
        if isinstance(spec, (int, float)):
            return DepthMeasurement(
                distance_m=float(spec),
                provenance=DistanceProvenance.SIMULATED,
                source=self._name,
                timestamp=now,
            )
        if isinstance(spec, tuple) and len(spec) == 2:
            kind, value = spec
            if kind == "MEASURED":
                return DepthMeasurement(
                    distance_m=float(value),
                    provenance=DistanceProvenance.MEASURED,
                    source=self._name,
                    timestamp=now,
                )
            if kind == "STALE":
                return DepthMeasurement(
                    distance_m=None,
                    provenance=DistanceProvenance.STALE,
                    source=self._name,
                    reason="scripted_stream_stale",
                    timestamp=now,
                )
            if kind == "OLD":
                return DepthMeasurement(
                    distance_m=float(value),
                    provenance=DistanceProvenance.SIMULATED,
                    source=self._name,
                    timestamp=now - self._stale_age_s,
                )
        raise ValueError(f"unsupported script spec {spec!r}")
