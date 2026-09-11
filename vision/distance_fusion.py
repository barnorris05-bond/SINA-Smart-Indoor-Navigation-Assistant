"""
vision/distance_fusion.py

Object-distance fusion (Phase C): attaches distance measurements from a
DepthProvider to DetectedObjects.

Design:

- Detection and depth are DECOUPLED: the detector never touches DepthAI.
  Fusion is the only place that calls the provider, per object, per frame.
- Provenance is carried end-to-end (DetectedObject.distance_provenance)
  and stamped into logs, so MOCK values are always identifiable.
- STALE handling: a measurement older than MEASUREMENT_STALE_S is
  rejected (and its provenance reported as STALE). Stale distance values
  are NEVER assigned — old data must not masquerade as current (unknown
  is never treated as safe).
- Unavailable objects keep distance=None with provenance UNAVAILABLE:
  distance-aware navigation then falls back to spatial-only behavior.

Note: fusion overwrites `distance` only for usable measurements
(MEASURED or SIMULATED); objects keep whatever distance they had if a
provider produced UNAVAILABLE this frame (e.g. legacy detectors that
pre-fill distance).
"""

import time
from typing import List, Optional

from depth.provider import DepthProvider, DepthMeasurement, DistanceProvenance
from vision.object_detector import DetectedObject
from config.depth_fusion import MEASUREMENT_STALE_S, ALLOW_SIMULATED_FOR_NAVIGATION
from utils.logger import logger


class DistanceFusion:
    """Fuses DepthProvider measurements into DetectedObjects."""

    def __init__(
        self,
        provider: DepthProvider,
        clock=time.time,
        stale_after_s: float = MEASUREMENT_STALE_S,
        allow_simulated: bool = ALLOW_SIMULATED_FOR_NAVIGATION,
    ) -> None:
        self._provider = provider
        self._clock = clock
        self._stale_after_s = stale_after_s
        self._allow_simulated = allow_simulated

        self._count = 0
        self._count_unavailable = 0
        self._count_stale = 0

    # ======================================================
    # Lifecycle passthrough
    # ======================================================

    def start(self) -> None:
        self._provider.start()

    def stop(self) -> None:
        self._provider.stop()

    def update(self) -> None:
        """Per-loop provider refresh (frame acquisition). Call before fuse()."""
        self._provider.update()

    # ======================================================
    # Fusion
    # ======================================================

    def fuse(
        self,
        detections: List[DetectedObject],
        frame_shape: tuple,
    ) -> List[DetectedObject]:
        """
        Attach distance to every detection (in place) and return them.

        frame_shape is (H, W) of the RGB frame the detections came from.
        Non-2D shapes are tolerated via shape[:2].
        """
        now = self._clock()
        for det in detections:
            measurement = self._provider.get_measurement(
                (det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2),
                frame_shape[:2],
                label=det.label,
                now=now,
            )
            self._attach(det, measurement, now)
        self._count += len(detections)
        return detections

    def _attach(
        self,
        det: DetectedObject,
        measurement: DepthMeasurement,
        now: float,
    ) -> None:
        # Defense in depth: reject measurements older than the window
        # even if a provider mislabels them.
        if (
            measurement.distance_m is not None
            and measurement.timestamp > 0
            and (now - measurement.timestamp) > self._stale_after_s
        ):
            measurement = DepthMeasurement(
                distance_m=None,
                provenance=DistanceProvenance.STALE,
                source=measurement.source,
                reason=f"too_old:{now - measurement.timestamp:.1f}s",
                timestamp=measurement.timestamp,
            )

        det.distance_source = measurement.source

        if measurement.usable and (
            measurement.provenance == DistanceProvenance.MEASURED
            or self._allow_simulated
        ):
            det.distance = measurement.distance_m
            det.distance_provenance = measurement.provenance.value
            return

        # Simulated value produced but disallowed for navigation:
        # stamp the object SIMULATED (honest labeling) and leave any
        # previous distance value untouched.
        if measurement.provenance == DistanceProvenance.SIMULATED:
            det.distance_provenance = DistanceProvenance.SIMULATED.value
            return

        # Provider reported STALE or UNAVAILABLE: keep the previous
        # distance value (legacy detectors may pre-fill one) and record
        # the fresh provenance so consumers never mistake it for current.
        if measurement.provenance in (
            DistanceProvenance.STALE,
            DistanceProvenance.UNAVAILABLE,
        ):
            det.distance_provenance = measurement.provenance.value
            if measurement.provenance == DistanceProvenance.STALE:
                self._count_stale += 1
                logger.info(
                    f"Fusion | {det.label} | distance stale "
                    f"({measurement.reason}); value withheld"
                )
            else:
                self._count_unavailable += 1
                logger.debug(
                    f"Fusion | {det.label} | distance unavailable "
                    f"({measurement.reason})"
                )

    # ======================================================
    # Telemetry
    # ======================================================

    def health(self) -> dict:
        return {
            "provider": self._provider.name,
            "provider_health": self._provider.health(),
            "fused_total": self._count,
            "unavailable_total": self._count_unavailable,
            "stale_total": self._count_stale,
            "allow_simulated": self._allow_simulated,
            "stale_after_s": self._stale_after_s,
        }
