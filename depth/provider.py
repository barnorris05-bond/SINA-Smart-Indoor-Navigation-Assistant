"""
depth/provider.py

The depth abstraction boundary (Phase B).

Everything downstream (fusion, navigation, audio) depends ONLY on the
DepthProvider interface and DepthMeasurement — never on DepthAI or on
mock internals. Replacing mock depth with real OAK-D stereo later
means swapping the provider instance (see depth/provider_factory.py),
with zero changes downstream.

Provenance model (requirement: never fake real measurements):

    MEASURED     - came from real stereo hardware (OakStereoDepthProvider)
    SIMULATED    - came from a mock provider (clearly marked everywhere)
    UNAVAILABLE  - no distance could be produced this frame
    STALE        - hardware previously produced measurements but the
                   stream has gone quiet; old values are NOT reused
                   silently (unknown is never treated as safe)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class DistanceProvenance(Enum):
    """How a distance value came to exist. Displayed via .value."""

    MEASURED = "MEASURED"
    SIMULATED = "SIMULATED"
    UNAVAILABLE = "UNAVAILABLE"
    STALE = "STALE"


# Bounding box as (x1, y1, x2, y2); frame shape as (H, W).
BBox = Tuple[int, int, int, int]
FrameShape = Tuple[int, int]


@dataclass(frozen=True)
class DepthMeasurement:
    """
    Outcome of one distance query for one object.

    distance_m is None whenever provenance is UNAVAILABLE or STALE
    (a stale value must never masquerade as a current measurement).
    """

    distance_m: Optional[float]
    provenance: DistanceProvenance
    source: str                     # provider id, e.g. "mock_scenario"
    reason: Optional[str] = None    # why unavailable/stale, for logs/tests
    timestamp: float = 0.0          # when the measurement was produced

    @property
    def usable(self) -> bool:
        """True when a distance value exists and may drive decisions."""
        return (
            self.distance_m is not None
            and self.provenance in (DistanceProvenance.MEASURED,
                                    DistanceProvenance.SIMULATED)
        )

    @property
    def provenance_str(self) -> str:
        return self.provenance.value


class DepthProvider(ABC):
    """
    Interface for per-object distance estimation.

    Implementations:
      - depth/mock_provider.ScenarioDepthProvider   (current, SIMULATED)
      - depth/mock_provider.BboxSyntheticProvider   (current, SIMULATED)
      - depth/oak_provider.OakStereoDepthProvider   (Phase E, MEASURED)

    Contract: get_measurement() is NON-BLOCKING and NEVER raises for
    expected conditions (no frame, bad geometry) — it returns a
    DepthMeasurement with the appropriate provenance instead.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier used in logs and measurements' source."""
        ...

    @abstractmethod
    def start(self) -> None:
        """Acquire resources (may be a no-op for mocks)."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Release resources (may be a no-op for mocks)."""
        ...

    @abstractmethod
    def get_measurement(
        self,
        bbox: BBox,
        frame_shape: FrameShape,
        label: Optional[str] = None,
        now: Optional[float] = None,
    ) -> DepthMeasurement:
        """
        Distance for the object at bbox inside a frame of frame_shape.

        label is optional object metadata: synthetic/mock providers use
        it (e.g. assumed real-world heights); stereo hardware ignores it.

        now is injectable (seconds) for deterministic tests; production
        callers omit it.
        """
        ...

    def update(self) -> None:
        """
        Per-loop refresh hook for providers that own a frame source.
        Call once per vision loop BEFORE fusing; default no-op.
        """
        return None

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Latest raw depth frame ((H, W) uint16 mm) if the provider has
        one; None otherwise. Mocks typically return None.
        """
        return None

    def health(self) -> dict:
        """Structured health snapshot for logging/UI/failure handling."""
        return {"provider": self.name, "ok": True}
