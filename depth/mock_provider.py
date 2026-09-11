"""
depth/mock_provider.py

MOCK depth providers (Phase B) — SIMULATED distances, clearly marked.

Purpose: let fusion / navigation / audio be developed and tested while
real stereo hardware validation is blocked (USB 2.0, pending Phase E).

!!! NOT REAL MEASUREMENTS !!!

Every DepthMeasurement carries provenance=SIMULATED and a source tag
naming the strategy, so no value can silently masquerade as a stereo
measurement. When real depth arrives (Phase E), this file is bypassed
entirely via depth/provider_factory.py.

Strategies (config/mock_depth.py chooses):
  scenario  - deterministic (label, region) table + label fallbacks.
              Best now: deterministic -> stable regression tests and
              full control of edge cases.
  bbox      - pinhole synthetic: d = (real_height * focal) / bbox_h.
              Depth-independence only holds for CENTER-lane objects;
              laterally offset objects are CLOSER than estimated -
              directional guidance only, never absolute.
"""

import time
from typing import Optional, Tuple

import numpy as np

from depth.provider import DepthProvider, DepthMeasurement, DistanceProvenance
from config.mock_depth import (
    SCENARIO_DISTANCES_M,
    LABEL_DEFAULT_DISTANCES_M,
    MOCK_MODE,
    BBOX_FOCAL_PX,
    BBOX_REAL_HEIGHT_M,
    BBOX_DEFAULT_HEIGHT_M,
    BBOX_MIN_M,
    BBOX_MAX_M,
)


def region_of(bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> str:
    """
    LEFT/CENTER/RIGHT by bbox center, matching DetectionManager's
    frame-thirds convention. Frame shape is (H, W).
    """
    x1, _, x2, _ = bbox
    frame_w = frame_shape[1]
    center_x = (x1 + x2) / 2.0
    if center_x < frame_w / 3:
        return "LEFT"
    if center_x < 2 * frame_w / 3:
        return "CENTER"
    return "RIGHT"


class ScenarioDepthProvider(DepthProvider):
    """
    Strategy A — deterministic scenario table lookup (RECOMMENDED NOW).

    Keyed by (label_lower, region); falls back to per-label defaults,
    then UNAVAILABLE. Never invents a number for unknown objects.
    """

    PROVENANCE = DistanceProvenance.SIMULATED

    def __init__(self, clock=time.time):
        # clock injectable for deterministic tests; production = time.time.
        self._clock = clock

    @property
    def name(self) -> str:
        return "mock_scenario"

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def get_measurement(
        self,
        bbox,
        frame_shape,
        label: Optional[str] = None,
        now: Optional[float] = None,
    ) -> DepthMeasurement:
        if not label:
            return self._unavailable("no_label")

        key = (label.lower(), region_of(bbox, frame_shape))
        distance = SCENARIO_DISTANCES_M.get(key)
        if distance is None:
            distance = LABEL_DEFAULT_DISTANCES_M.get(label.lower())
        if distance is None:
            return self._unavailable(f"no_scenario_entry:{label}")

        stamp = now if now is not None else self._clock()
        return DepthMeasurement(
            distance_m=distance,
            provenance=self.PROVENANCE,
            source=self.name,
            reason=None,
            timestamp=stamp,
        )

    def _unavailable(self, reason: str) -> DepthMeasurement:
        return DepthMeasurement(
            distance_m=None,
            provenance=DistanceProvenance.UNAVAILABLE,
            source=self.name,
            reason=reason,
            timestamp=self._clock(),
        )

    def health(self) -> dict:
        return {
            "provider": self.name,
            "ok": True,
            "note": "SIMULATED distances (mock) — NOT stereo measurements",
        }


class BboxSyntheticProvider(DepthProvider):
    """
    Strategy B — bounding-box pinhole synthetic distance.

    d = (assumed_real_height_m * FOCAL_PX) / bbox_height_px, clamped to
    the physical validity range. Values are SYNTHETIC ASSUMPTIONS
    (approximate focal, coarse average heights): usable for exercising
    continuous-distance code paths, NOT as range estimates.
    """

    PROVENANCE = DistanceProvenance.SIMULATED

    def __init__(self, clock=time.time):
        self._clock = clock

    @property
    def name(self) -> str:
        return "mock_bbox"

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def get_measurement(
        self,
        bbox,
        frame_shape,
        label: Optional[str] = None,
        now: Optional[float] = None,
    ) -> DepthMeasurement:
        x1, y1, x2, y2 = bbox
        bbox_h = y2 - y1

        if bbox_h <= 0:
            return self._unavailable("degenerate_bbox")

        real_height = BBOX_REAL_HEIGHT_M.get(
            (label or "").lower(), BBOX_DEFAULT_HEIGHT_M
        )
        distance = (real_height * BBOX_FOCAL_PX) / bbox_h
        distance = float(min(max(distance, BBOX_MIN_M), BBOX_MAX_M))

        stamp = now if now is not None else self._clock()
        return DepthMeasurement(
            distance_m=distance,
            provenance=self.PROVENANCE,
            source=self.name,
            reason=None,
            timestamp=stamp,
        )

    def _unavailable(self, reason: str) -> DepthMeasurement:
        return DepthMeasurement(
            distance_m=None,
            provenance=DistanceProvenance.UNAVAILABLE,
            source=self.name,
            reason=reason,
            timestamp=self._clock(),
        )

    def health(self) -> dict:
        return {
            "provider": self.name,
            "ok": True,
            "note": "SYNTHETIC bbox distances (mock) — NOT stereo measurements",
        }


class MockDepthProvider(DepthProvider):
    """
    Unified mock front-end. Delegates to the strategy chosen in
    config/mock_depth.py ("scenario" or "bbox"). Provenance is always
    SIMULATED — the fusion layer stamps objects with provenance_str,
    so MOCK data is visible in logs, UI and audio-facing state.
    """

    def __init__(self, mode: Optional[str] = None, clock=time.time):
        chosen = mode or MOCK_MODE
        if chosen == "bbox":
            self._impl: DepthProvider = BboxSyntheticProvider(clock=clock)
        elif chosen == "scenario":
            self._impl = ScenarioDepthProvider(clock=clock)
        else:
            raise ValueError(f"Unknown mock depth mode: {chosen!r}")

    @property
    def name(self) -> str:
        return self._impl.name

    def start(self) -> None:
        self._impl.start()

    def stop(self) -> None:
        self._impl.stop()

    def update(self) -> None:
        self._impl.update()

    def get_measurement(
        self,
        bbox,
        frame_shape,
        label: Optional[str] = None,
        now: Optional[float] = None,
    ) -> DepthMeasurement:
        return self._impl.get_measurement(bbox, frame_shape, label=label, now=now)

    def get_frame(self) -> Optional[np.ndarray]:
        return self._impl.get_frame()

    def health(self) -> dict:
        return self._impl.health()
