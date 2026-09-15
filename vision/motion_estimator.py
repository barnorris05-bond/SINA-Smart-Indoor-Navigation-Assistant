"""
vision/motion_estimator.py

MOTION / APPROACH ESTIMATION (Phase 6).

Answers, per tracked object: "what is this object's distance doing?"

    APPROACHING / STATIONARY / RECEDING / UNKNOWN
    + closing_rate_mps  (+ = distance decreasing = approaching)

Design (Phase 6 requirements):

- DEDICATED MODULE: motion logic lives here, not in ObjectTracker,
  DistanceFusion, DistanceNavigator or AudioManager. The estimator
  consumes tracker output; navigation still consumes distance only.
- HISTORY: per-track samples (timestamp, distance, provenance) keyed by
  track_id. Only VALID samples feed motion: MEASURED or SIMULATED with
  a positive distance. None / UNAVAILABLE / STALE never enter history
  and can never contaminate a rate.
- PROVENANCE HONESTY: the estimate's provenance is MEASURED only when
  every contributing sample is MEASURED; any SIMULATED input makes the
  estimate SIMULATED (simulated approach speed must never claim to be
  a real measurement). No valid samples -> UNAVAILABLE, rate None.
- FILTERING: two-half median rate. The estimation window is split into
  a first half and a second half; the median distance of each half is
  paired with the median timestamp of that half. Medians reject
  single-frame depth outliers (speckle, LRC failures, reflective
  glitches) where means would smear them, and half-center pairing has
  no lag bias for linear motion. Fully deterministic.
- NO OSCILLATION: hysteresis bands (config/motion.py). The estimator
  switches INTO APPROACHING/STATIONARY/RECEDING only when the rate is
  inside that state's band; dead-zone rates keep the previous state.
- GAP SAFETY: rates use only samples inside WINDOW_SPAN_S. After a
  long detection/staleness gap the window re-accumulates from scratch,
  so pre-gap evidence is never blended into a post-gap rate.
- NO RISK LOGIC: no navigation decisions here. Risk estimation is
  Phase 7. Timestamps come from the injectable clock so rates are
  honest m/s, not frame-count guesses.

Integration (main.py): after fusion.fuse(), before navigator.decide().
Purely additive annotation - navigation semantics unchanged.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from vision.object_detector import DetectedObject
from config.motion import (
    MIN_SAMPLES,
    MIN_OBSERVATION_S,
    WINDOW_SPAN_S,
    HISTORY_MAXLEN,
    MAX_HISTORY_AGE_FRAMES,
    APPROACH_RATE_MPS,
    RECEDE_RATE_MPS,
    STATIONARY_RATE_MPS,
)
from utils.logger import logger


class MotionState(str, Enum):
    """Temporal behavior of one tracked object's distance."""

    APPROACHING = "APPROACHING"
    STATIONARY = "STATIONARY"
    RECEDING = "RECEDING"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class _Sample:
    """One valid distance observation for one track."""

    timestamp: float
    distance_m: float
    provenance: str  # "MEASURED" | "SIMULATED"


@dataclass(frozen=True)
class MotionEstimate:
    """Per-object motion result stamped onto DetectedObject."""

    state: MotionState
    closing_rate_mps: Optional[float]  # + = approaching
    provenance: str                    # MEASURED | SIMULATED | UNAVAILABLE


class MotionEstimator:
    """
    Maintains per-track distance history and classifies motion.

    Usage: create ONCE for the application lifetime. Call update()
    once per frame with the fused detections (tracker already ran, so
    track_id is assigned; fusion already ran, so distance/provenance
    are stamped).

    Purely additive: returns the same DetectedObject instances with
    motion_state / closing_rate_mps / motion_provenance set. It never
    modifies distance, distance_provenance, track_id, label, bbox,
    priority or region, and never makes navigation decisions.
    """

    def __init__(self, clock=time.time) -> None:
        self._clock = clock
        self._history: Dict[int, List[_Sample]] = {}
        self._last_seen: Dict[int, int] = {}      # track_id -> estimator frame
        self._previous_state: Dict[int, MotionState] = {}  # hysteresis memory
        self._frame_index = 0

    # ======================================================
    # Public API
    # ======================================================

    def update(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        """
        Observe this frame's detections, update per-track histories and
        stamp motion fields onto every detection (in place).

        Tracks not present this frame keep their history (they may
        reappear within the tracker's MISSING window); history is
        purged after MAX_HISTORY_AGE_FRAMES consecutive absences.
        """
        self._frame_index += 1
        now = self._clock()

        self._purge_stale_tracks()

        for det in detections:
            track_id = det.track_id
            if track_id is None:
                # No tracker ran: no persistent identity, no history.
                # Honest per-frame UNKNOWN rather than invented motion.
                self._stamp(det, MotionEstimate(
                    MotionState.UNKNOWN, None, "UNAVAILABLE"))
                continue

            self._last_seen[track_id] = self._frame_index

            sample = self._valid_sample(det, now)
            if sample is not None:
                history = self._history.setdefault(track_id, [])
                history.append(sample)
                if len(history) > HISTORY_MAXLEN:
                    del history[: len(history) - HISTORY_MAXLEN]

            self._stamp(det, self._estimate(track_id, now))

        return detections

    def reset(self) -> None:
        """Clear all histories (e.g. on tracker reset / scene change)."""
        self._history.clear()
        self._last_seen.clear()
        self._previous_state.clear()

    def history_len(self, track_id: int) -> int:
        """Number of stored samples for a track (introspection/tests)."""
        return len(self._history.get(track_id, []))

    # ======================================================
    # Internals
    # ======================================================

    @staticmethod
    def _valid_sample(det: DetectedObject, now: float) -> Optional[_Sample]:
        """
        Extract a usable history sample or None.

        Only MEASURED/SIMULATED provenance with a positive distance is
        valid. UNAVAILABLE/STALE are never recorded - stale data must
        not silently become motion evidence.
        """
        if det.distance is None or det.track_id is None:
            return None
        if det.distance_provenance not in ("MEASURED", "SIMULATED"):
            return None
        if det.distance <= 0:
            return None
        return _Sample(timestamp=now, distance_m=float(det.distance),
                       provenance=det.distance_provenance)

    def _estimate(self, track_id: int, now: float) -> MotionEstimate:
        """
        Classify one track's motion from its recent history.

        Provenance rule: MEASURED requires ALL contributing samples to
        be MEASURED; any SIMULATED sample makes the estimate SIMULATED.
        """
        history = self._history.get(track_id, [])
        if len(history) < MIN_SAMPLES:
            return MotionEstimate(MotionState.UNKNOWN, None, "UNAVAILABLE")

        # Recent window only (time-bounded): a long detection gap means
        # the window re-accumulates before any rate is reported again.
        recent = [s for s in history if now - s.timestamp <= WINDOW_SPAN_S]
        if len(recent) < MIN_SAMPLES:
            return MotionEstimate(MotionState.UNKNOWN, None, "UNAVAILABLE")

        span = recent[-1].timestamp - recent[0].timestamp
        if span < MIN_OBSERVATION_S:
            return MotionEstimate(MotionState.UNKNOWN, None, "UNAVAILABLE")

        provenance = (
            "MEASURED" if all(s.provenance == "MEASURED" for s in recent)
            else "SIMULATED"
        )

        rate = self._half_median_rate(recent)
        if rate is None:
            return MotionEstimate(MotionState.UNKNOWN, None, provenance)

        state = self._classify(rate, track_id)
        return MotionEstimate(state, rate, provenance)

    @staticmethod
    def _half_median_rate(samples: List[_Sample]) -> Optional[float]:
        """
        Two-half median rate estimate.

        Split the window into a first half and a second half (middle
        sample dropped for odd sizes). rate = (median_d_first_half -
        median_d_second_half) / (median_t_second - median_t_first).

        - Median halves reject single-frame depth outliers.
        - Half-center timestamps pair without lag bias for linear
          motion (unlike raw endpoints vs a trailing median).
        - Returns None if the two halves share a timestamp (degenerate).
        """
        n = len(samples)
        half = n // 2
        first_half = samples[:half]
        second_half = samples[n - half:]

        def median(values: List[float]) -> float:
            values = sorted(values)
            m = len(values)
            return values[m // 2] if m % 2 == 1 \
                else (values[m // 2 - 1] + values[m // 2]) / 2.0

        t_first = median([s.timestamp for s in first_half])
        t_second = median([s.timestamp for s in second_half])
        dt = t_second - t_first
        if dt <= 0:
            return None

        d_first = median([s.distance_m for s in first_half])
        d_second = median([s.distance_m for s in second_half])
        # Distance decreasing -> positive closing rate (approaching).
        return (d_first - d_second) / dt

    def _classify(self, rate: float, track_id: int) -> MotionState:
        """
        Band classification with hysteresis dead zones.

        Switch INTO a state only when the rate is inside its band;
        dead-zone rates (between the STATIONARY and APPROACH/RECEDE
        bands) keep the previous state - no flicker.
        """
        previous = self._previous_state.get(track_id)

        if -STATIONARY_RATE_MPS <= rate <= STATIONARY_RATE_MPS:
            return MotionState.STATIONARY
        if rate >= APPROACH_RATE_MPS:
            return MotionState.APPROACHING
        if rate <= -RECEDE_RATE_MPS:
            return MotionState.RECEDING

        return previous or MotionState.STATIONARY

    def _stamp(self, det: DetectedObject, estimate: MotionEstimate) -> None:
        """Remember the state (for hysteresis) and stamp fields."""
        if det.track_id is not None:
            self._previous_state[det.track_id] = estimate.state
        det.motion_state = estimate.state.value
        det.closing_rate_mps = estimate.closing_rate_mps
        det.motion_provenance = estimate.provenance

    def _purge_stale_tracks(self) -> None:
        """Drop history for tracks unseen for MAX_HISTORY_AGE_FRAMES."""
        expired = [
            tid for tid, seen in self._last_seen.items()
            if self._frame_index - seen > MAX_HISTORY_AGE_FRAMES
        ]
        for tid in expired:
            self._history.pop(tid, None)
            self._last_seen.pop(tid, None)
            self._previous_state.pop(tid, None)
            logger.debug(f"Motion history purged for track {tid}")
