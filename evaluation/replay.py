"""
evaluation/replay.py

OFFLINE REPLAY (deterministic, hardware-free).

Drives the REAL decision pipeline — DetectionManager → ObjectTracker →
DistanceFusion → MotionEstimator → RiskEstimator → TemporalNavigator —
over scripted per-frame scenes and a scripted depth provider. No OAK-D,
no YOLO inference, no audio, no GUI.

    ReplayConfig(scene_id, frame_rate, ...)
    ReplayRunner(config).run(frames)

where each ScriptedFrame is {detections: [(label, bbox, conf), ...],
distances: {label: provider script spec}}. Distance specs are passed to
evaluation/injection.ScriptedDepthProvider (None, float, ("MEASURED", v),
("STALE", v), ("OLD", v)).

Determinism: a synthetic clock advances by exactly 1/frame_rate s per
frame and is shared by fusion + motion so one frame = one timestamp;
component order mirrors main.py exactly. The same frames always
produce the same trace, records and decisions. The runner is stateless
between runs (every component is constructed fresh in run()).

Outputs:
    records   - MeasurementRecord per (frame, object); provenance is
                taken from the REAL DistanceFusion stamping, never
                re-derived or altered
    events    - parsed EventLogger per-frame trace
    decisions - NavigationDecision per frame
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from vision.object_detector import DetectedObject, BoundingBox
from vision.detection_manager import DetectionManager
from vision.object_tracker import ObjectTracker
from vision.distance_fusion import DistanceFusion
from vision.motion_estimator import MotionEstimator
from navigation.risk_estimator import RiskEstimator
from navigation.distance_navigator import DistanceNavigator
from navigation.navigator import NavigationDecision
from navigation.temporal_navigator import TemporalNavigator

from evaluation.record import MeasurementRecord, SCHEMA_VERSION
from evaluation.injection import ScriptedDepthProvider, ScriptSpec
from evaluation.event_log import EventLogger
from config.temporal_navigation import (
    DEESCALATION_CONFIRM_FRAMES,
    ACTION_CHANGE_CONFIRM_FRAMES,
    CRITICAL_OVERRIDE,
    CRITICAL_RISK_FORCE_STOP,
)

# (label, bbox(x1, y1, x2, y2), confidence)
DetectionSpec = Tuple[str, Tuple[int, int, int, int], float]


class _ScriptClock:
    """
    Synthetic clock shared by fusion + motion: set() before each frame
    so every component sees the SAME frame timestamp; advance() moves
    to the next frame boundary. Injectable-clock contract identical to
    the production layers (a zero-arg callable returning seconds).
    """

    def __init__(self, start: float) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def set(self, t: float) -> None:
        self.t = t


@dataclass
class ScriptedFrame:
    """One synthetic scene: objects plus their distance script."""

    detections: List[DetectionSpec]
    distances: Dict[str, ScriptSpec] = field(default_factory=dict)
    frame_width: int = 640
    frame_height: int = 400


@dataclass
class ReplayConfig:
    """Replay parameters (deterministic by construction)."""

    scene_id: str = "replay"
    frame_rate: float = 15.0
    start_timestamp: float = 1000.0
    # Temporal-navigation policy knobs (development/test defaults from
    # config/temporal_navigation.py; exposed so replay experiments can
    # exercise alternative confirmation policies deterministically).
    deescalation_confirm_frames: int = DEESCALATION_CONFIRM_FRAMES
    action_change_confirm_frames: int = ACTION_CHANGE_CONFIRM_FRAMES
    critical_override: bool = CRITICAL_OVERRIDE
    critical_risk_force_stop: bool = CRITICAL_RISK_FORCE_STOP


@dataclass
class ReplayResult:
    """Everything one replay produced (trace, records, decisions)."""

    scene_id: str
    records: List[MeasurementRecord]
    events: List[dict]                    # parsed EventLogger events
    decisions: List[NavigationDecision]
    frames_run: int


class ReplayRunner:
    """Runs scripted frames through the real decision pipeline."""

    def __init__(self, config: Optional[ReplayConfig] = None) -> None:
        self.config = config or ReplayConfig()

    # ------------------------------------------------------
    # Public API
    # ------------------------------------------------------

    def run(self, frames: List[ScriptedFrame]) -> ReplayResult:
        if not frames:
            raise ValueError("replay needs at least one ScriptedFrame")
        cfg = self.config

        provider = ScriptedDepthProvider(
            [f.distances for f in frames],
            name=f"scripted:{cfg.scene_id}",
        )
        clock = _ScriptClock(cfg.start_timestamp)
        fusion = DistanceFusion(provider, clock=clock)
        tracker = ObjectTracker()
        motion = MotionEstimator(clock=clock)
        risk = RiskEstimator()
        navigator = TemporalNavigator(
            DistanceNavigator(),
            deescalation_confirm_frames=cfg.deescalation_confirm_frames,
            action_change_confirm_frames=cfg.action_change_confirm_frames,
            critical_override=cfg.critical_override,
            critical_risk_force_stop=cfg.critical_risk_force_stop,
        )
        detection_manager = DetectionManager()
        events = EventLogger(sink=None)  # in-memory; file sink not needed here

        records: List[MeasurementRecord] = []
        decisions: List[NavigationDecision] = []
        dt = 1.0 / cfg.frame_rate

        try:
            fusion.start()
            for frame_id, frame in enumerate(frames):
                clock.set(cfg.start_timestamp + frame_id * dt)

                # 1) raw detections -> enrichment (same order as main.py)
                raw = [
                    DetectedObject(
                        label=label,
                        confidence=conf,
                        bbox=BoundingBox(x1, y1, x2, y2),
                    )
                    for (label, (x1, y1, x2, y2), conf) in frame.detections
                ]
                detections = detection_manager.process(raw, frame.frame_width)

                # 2) tracking
                detections = tracker.update(detections)

                # 3) distance fusion (real provenance stamping)
                fusion.update()
                detections = fusion.fuse(
                    detections, (frame.frame_height, frame.frame_width))

                # 4) motion estimation
                detections = motion.update(detections)

                # 5) risk estimation
                detections = risk.update(detections)

                # 6) temporal navigation decision
                decision = navigator.decide(detections)
                decisions.append(decision)

                # 7) trace + records
                events.log_frame(frame_id, clock.t, detections, decision)
                records.extend(
                    self._records(cfg.scene_id, frame_id, clock.t, detections))
        finally:
            fusion.stop()
            events.close()

        return ReplayResult(
            scene_id=cfg.scene_id,
            records=records,
            events=events.parsed(),
            decisions=decisions,
            frames_run=len(frames),
        )

    # ------------------------------------------------------
    # Internals
    # ------------------------------------------------------

    @staticmethod
    def _records(
        scene_id: str,
        frame_id: int,
        ts: float,
        detections: List[DetectedObject],
    ) -> List[MeasurementRecord]:
        out: List[MeasurementRecord] = []
        for det in detections:
            out.append(MeasurementRecord(
                schema_version=SCHEMA_VERSION,
                scene_id=scene_id,
                frame_id=frame_id,
                timestamp=ts,
                track_id=det.track_id,
                label=det.label,
                confidence=det.confidence,
                bbox=(det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2),
                region=det.region,
                predicted_distance_m=det.distance,
                distance_provenance=det.distance_provenance,
                distance_source=det.distance_source,
                ground_truth_distance_m=None,
            ))
        return out
