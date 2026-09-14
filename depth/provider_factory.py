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
from config.camera import ENABLE_STEREO_HARDWARE
from utils.logger import logger
from depth.provider import DepthProvider
from depth.mock_provider import MockDepthProvider
from vision.distance_fusion import DistanceFusion


def get_depth_provider(camera_manager=None) -> DepthProvider:
    """
    Build the configured DepthProvider.

    camera_manager is required for the real stereo backend; it is
    ignored (may be None) for mocks.

    Pairing rule (Phase 9): USE_MOCK_DEPTH=False selects the real OAK
    stereo provider and requires ENABLE_STEREO_HARDWARE=True — if the
    hardware gate is off the provider is still built (so the factory
    remains the single swap point) but every measurement will honestly
    report UNAVAILABLE; that mismatch is reported LOUDLY here, never
    silently.
    """
    if USE_MOCK_DEPTH:
        return MockDepthProvider()

    if not ENABLE_STEREO_HARDWARE:
        logger.warning(
            "CONFIG MISMATCH: USE_MOCK_DEPTH=False but "
            "ENABLE_STEREO_HARDWARE=False — stereo branch will never be "
            "built; all distances will report UNAVAILABLE (RGB-only). "
            "Enable the stereo gate (see tests/test_depth_stream.py) or "
            "set USE_MOCK_DEPTH=True."
        )

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
