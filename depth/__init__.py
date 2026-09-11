"""
SINA depth subsystem (Phases 2 / B-E).

Public API:
- DepthProvider / DepthMeasurement / DistanceProvenance: the provider
  abstraction everything downstream depends on (never on DepthAI).
- MockDepthProvider: SIMULATED strategies while stereo is blocked.
- validity helpers: robust measurement for the real stereo path.
Frame acquisition lives in camera/ (DepthAI boundary).
"""

from depth.provider import (
    DepthProvider,
    DepthMeasurement,
    DistanceProvenance,
)
from depth.mock_provider import (
    MockDepthProvider,
    ScenarioDepthProvider,
    BboxSyntheticProvider,
    region_of,
)
from depth.validity import (
    DepthStats,
    validate_depth_frame,
    invalid_mask,
    measure_region,
    central_roi,
)

__all__ = [
    "DepthProvider",
    "DepthMeasurement",
    "DistanceProvenance",
    "MockDepthProvider",
    "ScenarioDepthProvider",
    "BboxSyntheticProvider",
    "region_of",
    "DepthStats",
    "validate_depth_frame",
    "invalid_mask",
    "measure_region",
    "central_roi",
]
