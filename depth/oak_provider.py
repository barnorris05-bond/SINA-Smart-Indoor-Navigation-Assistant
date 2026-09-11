"""
depth/oak_provider.py

Real OAK-D stereo depth provider (Phase E implementation, hardware
validation pending).

Wraps the existing CameraManager depth stream (which handles pipeline
construction, non-blocking retrieval, warmup and health tracking) and
adapts it to the DepthProvider interface. Distance computation reuses
depth/validity.py: central-ROI median, minimum samples, valid ratio —
the exact statistics documented in Phase 2.

Provenance is always MEASURED when a value is produced; when the
stream is down, stale, or a region cannot be measured, provenance is
STALE / UNAVAILABLE — never a fabricated number.

Hardware caveat (documented, not hidden): stereo hardware validation
is currently BLOCKED (USB 2.0 link; a custom stereo pipeline produced
X_LINK_ERROR and device reconnection). This provider is implemented
against the same CameraManager path used by tests/test_depth_stream.py
and must be treated as UNVALIDATED until that test PASSES on hardware.
"""

import time
from typing import Optional

import numpy as np

from depth.provider import DepthProvider, DepthMeasurement, DistanceProvenance
from depth.validity import measure_region


class OakStereoDepthProvider(DepthProvider):
    """
    DepthProvider backed by CameraManager's aligned stereo-depth stream.

    camera_manager must expose: get_depth_frame() (or None),
    get_depth_health() and is_depth_available(). It is referenced, not
    owned: lifecycle (start/stop of the device) stays with CameraManager.
    """

    PROVENANCE = DistanceProvenance.MEASURED

    def __init__(self, camera_manager, clock=time.time) -> None:
        self._camera = camera_manager
        self._clock = clock
        self._frame: Optional[np.ndarray] = None
        self._frame_at: float = 0.0

    @property
    def name(self) -> str:
        return "oak_stereo"

    def start(self) -> None:
        # Device lifecycle belongs to CameraManager (main.py starts it).
        return None

    def stop(self) -> None:
        self._frame = None
        self._frame_at = 0.0

    def update(self) -> None:
        """Pull the newest depth frame (non-blocking) once per loop."""
        frame = self._camera.get_depth_frame()
        if frame is not None:
            self._frame = frame
            self._frame_at = self._clock()

    def get_measurement(
        self,
        bbox,
        frame_shape,
        label: Optional[str] = None,   # stereo ignores labels
        now: Optional[float] = None,
    ) -> DepthMeasurement:
        stamp = now if now is not None else self._clock()

        if not self._camera.is_depth_available():
            return DepthMeasurement(
                distance_m=None,
                provenance=DistanceProvenance.UNAVAILABLE,
                source=self.name,
                reason=f"depth_stream:{self._camera.depth_status}",
                timestamp=stamp,
            )

        if self._frame is None:
            return DepthMeasurement(
                distance_m=None,
                provenance=DistanceProvenance.UNAVAILABLE,
                source=self.name,
                reason="no_frame_yet",
                timestamp=stamp,
            )

        if stamp - self._frame_at > 0.5:
            # CameraManager reports stream health; a frame we haven't
            # refreshed for 0.5s (>= several frame periods at 15 FPS)
            # is treated as stale so old values never drive decisions.
            return DepthMeasurement(
                distance_m=None,
                provenance=DistanceProvenance.STALE,
                source=self.name,
                reason=f"frame_age:{stamp - self._frame_at:.2f}s",
                timestamp=self._frame_at,
            )

        # Detections are in RGB-frame coordinates; depth is aligned to
        # CAM_A so sizes normally match, but scale defensively if the
        # depth frame dimensions ever differ.
        x1, y1, x2, y2 = bbox
        frame_h, frame_w = frame_shape[:2]
        depth_h, depth_w = self._frame.shape[:2]
        if (depth_h, depth_w) != (frame_h, frame_w):
            sx = depth_w / float(frame_w)
            sy = depth_h / float(frame_h)
            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)

        stats = measure_region(self._frame, (x1, y1, x2, y2))
        if not stats.valid:
            return DepthMeasurement(
                distance_m=None,
                provenance=DistanceProvenance.UNAVAILABLE,
                source=self.name,
                reason=f"region:{stats.reason}",
                timestamp=self._frame_at,
            )

        return DepthMeasurement(
            distance_m=stats.distance_m,
            provenance=self.PROVENANCE,
            source=self.name,
            reason=None,
            timestamp=self._frame_at,
        )

    def get_frame(self) -> Optional[np.ndarray]:
        return self._frame

    def health(self) -> dict:
        health = dict(self._camera.get_depth_health())
        health["provider"] = self.name
        return health
