"""
depth/provider_factory.py

Single place where the depth backend is chosen.

Phase B..D (stereo blocked):  USE_MOCK_DEPTH = True  -> MockDepthProvider
Phase E   (USB 3.x obtained,  test_depth_stream.py PASS):
    set USE_MOCK_DEPTH = False -> OakStereoDepthProvider

Nothing downstream (fusion, navigation, audio) changes either way —
they depend only on the DepthProvider interface.
"""

from typing import Optional

from config.mock_depth import USE_MOCK_DEPTH
from depth.provider import DepthProvider
from depth.mock_provider import MockDepthProvider
from vision.distance_fusion import DistanceFusion


def get_depth_provider(camera_manager=None) -> DepthProvider:
    """
    Build the configured DepthProvider.

    camera_manager is required for the real stereo backend; it is
    ignored (may be None) for mocks.
    """
    if USE_MOCK_DEPTH:
        return MockDepthProvider()

    if camera_manager is None:
        raise ValueError(
            "USE_MOCK_DEPTH is False but no camera_manager was provided "
            "for the OAK stereo backend."
        )

    # Imported here so mock-mode runs never import the DepthAI-adjacent
    # module unnecessarily.
    from depth.oak_provider import OakStereoDepthProvider

    return OakStereoDepthProvider(camera_manager)


def create_distance_fusion(camera_manager=None) -> DistanceFusion:
    """Convenience: provider + fusion configured from config files."""
    provider = get_depth_provider(camera_manager)
    return DistanceFusion(provider)
