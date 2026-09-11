"""
vision/object_tracker.py

Deterministic object tracking across frames (Phase 5).

Assigns persistent, monotonically increasing track_id values to
detections produced by the existing vision pipeline:

    YOLODetector -> DetectionManager -> ObjectTracker -> DistanceFusion

Design constraints (Phase 5 requirements):

- HARDWARE-INDEPENDENT: imports no DepthAI, no YOLO, no OpenCV.
  Operates purely on DetectedObject values.
- ORDER-INDEPENDENT: detection list order never influences identity —
  matching is geometric/label based (DetectionManager already sorts by
  priority, so list position is meaningless anyway).
- ADDITIVE: the tracker only sets `track_id`. It never rewrites label,
  confidence, bbox, region, priority, distance, distance_provenance or
  distance_source — those belong to the existing systems. Provenance
  (SIMULATED/MEASURED/STALE/UNAVAILABLE) passes through untouched.
- ONE-TO-ONE: a track matches at most one detection per frame and a
  detection matches at most one track.
- NO NAVIGATION LOGIC: the tracker only answers "which detection
  corresponds to which persistent object?".

Algorithm: deterministic greedy two-pass matching (documented choice —
see config/tracking.py and the CHANGELOG for why ByteTrack/BoT-SORT
were deferred):

  Pass 1: label-compatible pairs by descending IoU (box overlap).
  Pass 2: remaining pairs by ascending centroid distance.

  A candidate pair is valid only if label matches AND (IoU >= MIN_IOU
  OR centroid distance <= MAX_CENTROID_DISTANCE).

Complexity: O(T x D) per frame (T tracks, D detections) with a greedy
sort over candidate pairs — comfortably fast for 15 FPS.

Lifecycle: NEW (created this frame) -> ACTIVE (matched) -> MISSING
(temporarily undetected, retained up to MAX_MISSED_FRAMES) -> EXPIRED
(removed). MISSING tracks can re-match and become ACTIVE again.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from vision.object_detector import DetectedObject, BoundingBox
from config.tracking import MAX_MISSED_FRAMES, MIN_IOU, MAX_CENTROID_DISTANCE
from utils.logger import logger


class TrackState(Enum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    MISSING = "MISSING"
    EXPIRED = "EXPIRED"


@dataclass
class Track:
    """Internal track state for one persistent physical object."""

    track_id: int
    label: str
    bbox: BoundingBox
    center_x: float
    center_y: float
    age: int = 0              # frames since creation (including this one)
    hits: int = 0             # total matched detections
    missed_frames: int = 0    # consecutive unmatched frames
    state: TrackState = TrackState.NEW
    last_seen_frame: int = 0  # frame index of the last match


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union of two bounding boxes."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def centroid_distance(a: BoundingBox, b: BoundingBox) -> float:
    """Euclidean distance between bounding-box centers."""
    ax, ay = (a.x1 + a.x2) / 2.0, (a.y1 + a.y2) / 2.0
    bx, by = (b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


class ObjectTracker:
    """
    Persistent multi-object tracker over DetectedObject streams.

    Usage: create ONCE for the application lifetime and call update()
    with each frame's enriched detections. Detection order does not
    affect identity. Returned objects are the SAME instances passed in,
    with track_id assigned (fields otherwise untouched).
    """

    def __init__(
        self,
        max_missed_frames: int = MAX_MISSED_FRAMES,
        min_iou: float = MIN_IOU,
        max_centroid_distance: float = MAX_CENTROID_DISTANCE,
    ) -> None:
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be >= 0")
        if min_iou < 0.0 or min_iou > 1.0:
            raise ValueError("min_iou must be within [0, 1]")
        if max_centroid_distance < 0:
            raise ValueError("max_centroid_distance must be >= 0")

        self._max_missed = max_missed_frames
        self._min_iou = min_iou
        self._max_centroid = max_centroid_distance

        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._frame_index = 0

    # ======================================================
    # Public API
    # ======================================================

    def update(
        self,
        detections: List[DetectedObject],
        timestamp: Optional[float] = None,   # reserved: logging/tests only
    ) -> List[DetectedObject]:
        """
        Match detections to existing tracks, assign track_ids, update
        lifecycle state, and return the input list (objects mutated in
        place with track_id set).

        timestamp is accepted for API compatibility/future use; the
        tracker currently operates on frame counts.
        """
        self._frame_index += 1
        frame = self._frame_index

        self._expire_missing()
        assignments, matched_det_indices = self._match(detections)

        for det, track in assignments:
            det.track_id = track.track_id

        self._update_track_states(assignments, matched_det_indices, detections, frame)
        return detections

    def reset(self) -> None:
        """Clear all tracks; the next detection starts a fresh ID."""
        self._tracks.clear()
        # IDs stay monotonic after reset: no ID reuse, simpler logs/tests.

    def active_tracks(self) -> List[Track]:
        """Snapshot of non-expired tracks (tests/introspection)."""
        return list(self._tracks.values())

    def track_count(self) -> int:
        return len(self._tracks)

    # ======================================================
    # Matching
    # ======================================================

    def _match(
        self, detections: List[DetectedObject]
    ) -> Tuple[List[Tuple[DetectedObject, Track]], set]:
        """
        Deterministic greedy one-to-one assignment.

        Returns (assignments, matched_detection_indices).

        Candidate pairs (label-compatible AND geometrically plausible)
        are considered pass-1 (IoU, descending) then pass-2 (centroid
        distance, ascending). Greedy best-first: each pair is taken only
        if both members are still free — this enforces one-to-one.
        """
        assignments: List[Tuple[DetectedObject, Track]] = []
        used_tracks: set = set()
        used_dets: set = set()

        # ---- Pass 1: IoU (box overlap), best first -----------------
        pass1: List[Tuple[float, int, int]] = []
        for ti, track in self._tracks.items():
            for di, det in enumerate(detections):
                if det.label.lower() != track.label.lower():
                    continue
                overlap = iou(track.bbox, det.bbox)
                if overlap >= self._min_iou:
                    pass1.append((overlap, ti, di))
        # Descending IoU; (ti, di) tiebreak keeps determinism.
        pass1.sort(key=lambda t: (-t[0], t[1], t[2]))

        for _, ti, di in pass1:
            if ti in used_tracks or di in used_dets:
                continue
            assignments.append((detections[di], self._tracks[ti]))
            used_tracks.add(ti)
            used_dets.add(di)

        # ---- Pass 2: centroid distance, nearest first ---------------
        pass2: List[Tuple[float, int, int]] = []
        for ti, track in self._tracks.items():
            if ti in used_tracks:
                continue
            for di, det in enumerate(detections):
                if di in used_dets:
                    continue
                if det.label.lower() != track.label.lower():
                    continue
                dist = centroid_distance(track.bbox, det.bbox)
                if dist <= self._max_centroid:
                    pass2.append((dist, ti, di))
        pass2.sort(key=lambda t: (t[0], t[1], t[2]))

        for _, ti, di in pass2:
            if ti in used_tracks or di in used_dets:
                continue
            assignments.append((detections[di], self._tracks[ti]))
            used_tracks.add(ti)
            used_dets.add(di)

        return assignments, used_dets

    # ======================================================
    # Lifecycle
    # ======================================================

    def _expire_missing(self) -> None:
        """Drop tracks that exceeded MAX_MISSED_FRAMES."""
        expired = [
            tid for tid, t in self._tracks.items()
            if t.missed_frames > self._max_missed
        ]
        for tid in expired:
            track = self._tracks.pop(tid)
            track.state = TrackState.EXPIRED
            logger.debug(f"Track {tid} expired: {track.label}")

    def _update_track_states(
        self,
        assignments: List[Tuple[DetectedObject, Track]],
        matched_det_indices: set,
        detections: List[DetectedObject],
        frame: int,
    ) -> None:
        matched_track_ids = set()
        created_track_ids = set()

        for det, track in assignments:
            track.age += 1
            track.bbox = det.bbox
            track.center_x = (det.bbox.x1 + det.bbox.x2) / 2.0
            track.center_y = (det.bbox.y1 + det.bbox.y2) / 2.0
            track.hits += 1
            track.missed_frames = 0
            # NEW is the birth state; any successful match (including
            # the return of a MISSING track) makes the track ACTIVE.
            track.state = TrackState.ACTIVE
            track.last_seen_frame = frame
            matched_track_ids.add(track.track_id)
            logger.debug(f"Track {track.track_id} matched: {track.label}")

        # Unmatched detections -> NEW tracks (monotonic IDs, never reused).
        for di, det in enumerate(detections):
            if di in matched_det_indices:
                continue
            new_track = Track(
                track_id=self._next_id,
                label=det.label.lower(),
                bbox=det.bbox,
                center_x=(det.bbox.x1 + det.bbox.x2) / 2.0,
                center_y=(det.bbox.y1 + det.bbox.y2) / 2.0,
                age=1,
                hits=1,
                missed_frames=0,
                state=TrackState.NEW,
                last_seen_frame=frame,
            )
            det.track_id = new_track.track_id
            self._tracks[new_track.track_id] = new_track
            created_track_ids.add(new_track.track_id)
            self._next_id += 1
            logger.debug(f"Track {new_track.track_id} created: {new_track.label}")

        # Unmatched existing tracks -> MISSING (tracks created this
        # frame are NOT missing — this is their first frame of life).
        for tid, track in self._tracks.items():
            if tid in matched_track_ids or tid in created_track_ids:
                continue
            track.age += 1
            track.missed_frames += 1
            if track.state in (TrackState.NEW, TrackState.ACTIVE):
                track.state = TrackState.MISSING
            logger.debug(
                f"Track {tid} missing: {track.missed_frames} frame(s)"
            )
