"""
calibration/capture.py

CONTROLLED CAPTURE STATE MACHINE (Phase 10 data-collection protocol).

One Capture = one planned slot of the experiment plan: a target
distance and repetition (optionally per scene), e.g.

    target 2.0 m, repetition 3  ->  Capture(PLANNED -> ... -> COMPLETED)

Rules enforced here (milestone §5-§12):

- DETERMINISTIC STATE MACHINE (§5): PLANNED -> ACTIVE -> {COMPLETED,
  INCOMPLETE, ABORTED}; PLANNED -> ABORTED. Terminal states accept no
  further transitions; invalid transitions raise CaptureStateError.
  An interrupted capture is never treated as completed.
- RAW/DERIVED SEPARATION (§9): raw MeasurementRecords are stored
  verbatim and never mutated. Records with attached ground truth are a
  DERIVED view (derived_records()); the raw list always reflects the
  untouched captures.
- TARGET vs GROUND TRUTH (§12): target_distance_m (the plan's nominal
  slot) and ground truth (the operator's observed reference) are
  distinct fields, never merged or substituted.
- GROUND-TRUTH AUDIT (§11): every attachment appends a
  GroundTruthEntry recording which observation identity received which
  value, from which source/operator note. Uncertainty is stored only
  when the operator explicitly supplies one — never invented.
- INCOMPLETE vs DISCARD (§8): a capture that fails its completeness
  rule (documented below) is marked INCOMPLETE, never silently
  discarded or converted to a success.

Completeness rule (documented per §8; all checked at complete()):
  - at least min_observations raw records attached (default 1)
  - every record: identity known (scene_id non-empty, frame_id >= 0)
  - every record: provenance present (project vocabulary)
  - every record: finite, non-negative timestamp and valid frame id
  - ground truth is NOT required for capture completion — truth is a
    separate, later workflow step (its absence is reported by the
    experiment-level completion check, not silently at capture level)
"""

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from evaluation.record import MeasurementRecord, VALID_PROVENANCES

from calibration.session import ExperimentSession


class CaptureStateError(ValueError):
    """Raised on invalid capture state transitions."""


class CaptureNotActiveError(CaptureStateError):
    """Raised when records are added to a non-ACTIVE capture."""


# --------------------------------------------------------------
# States
# --------------------------------------------------------------

PLANNED = "PLANNED"
ACTIVE = "ACTIVE"
COMPLETED = "COMPLETED"
INCOMPLETE = "INCOMPLETE"
ABORTED = "ABORTED"

_TERMINAL = {COMPLETED, INCOMPLETE, ABORTED}

#: The complete, deterministic transition table.
VALID_TRANSITIONS: Dict[str, Tuple[str, ...]] = {
    PLANNED: (ACTIVE, ABORTED),
    ACTIVE: (COMPLETED, INCOMPLETE, ABORTED),
    COMPLETED: (),
    INCOMPLETE: (),
    ABORTED: (),
}


def _check_transition(current: str, new: str) -> None:
    if current not in VALID_TRANSITIONS:
        raise CaptureStateError(f"unknown capture state {current!r}")
    if new not in _TERMINAL | {ACTIVE, PLANNED}:
        raise CaptureStateError(f"unknown capture state {new!r}")
    if new == current:
        raise CaptureStateError(
            f"capture is already {current!r}; no-op transition rejected"
        )
    if new not in VALID_TRANSITIONS[current]:
        raise CaptureStateError(
            f"invalid transition {current} -> {new}; allowed: "
            f"{VALID_TRANSITIONS[current] or '(none - terminal state)'}"
        )


# --------------------------------------------------------------
# Ground-truth audit entry (§11)
# --------------------------------------------------------------

@dataclass(frozen=True)
class GroundTruthEntry:
    """
    Audit record for one ground-truth attachment (§11).

    uncertainty_m is stored ONLY when the operator explicitly supplies
    it (e.g. "tape measure ±5 mm") — never invented here.
    """

    scene_id: str
    frame_id: int
    track_id: Optional[int]
    ground_truth_distance_m: float
    source: Optional[str] = None       # operator/method note
    uncertainty_m: Optional[float] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "track_id": self.track_id,
            "ground_truth_distance_m": self.ground_truth_distance_m,
            "source": self.source,
            "uncertainty_m": self.uncertainty_m,
            "note": self.note,
        }


# --------------------------------------------------------------
# Capture
# --------------------------------------------------------------

@dataclass
class Capture:
    """
    One planned experiment slot with an explicit lifecycle.

    Repetitions are first-class: each Capture is (target_distance_m,
    repetition) — repeating a slot creates a NEW Capture; previous
    repetitions are never overwritten (§7).
    """

    target_distance_m: float
    repetition: int
    scene_id: Optional[str] = None
    session: Optional[ExperimentSession] = None
    notes: Optional[str] = None

    state: str = PLANNED
    records: List[MeasurementRecord] = field(default_factory=list)
    ground_truth_log: List[GroundTruthEntry] = field(default_factory=list)
    start_time: Optional[str] = None   # ISO-8601 string or None (§4)
    end_time: Optional[str] = None
    # free-form operator-supplied context; never fabricated by code
    metadata: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------

    def start(self, start_time: Optional[str] = None) -> "Capture":
        """PLANNED -> ACTIVE. start_time is operator-supplied or None."""
        _check_transition(self.state, ACTIVE)
        self.state = ACTIVE
        self.start_time = start_time
        return self

    def complete(self, end_time: Optional[str] = None,
                 min_observations: int = 1) -> "Capture":
        """
        ACTIVE -> COMPLETED (if the documented completeness rule holds)
        or ACTIVE -> INCOMPLETE (rule failed; capture is kept and
        visible, never discarded).
        """
        _check_transition(self.state, COMPLETED)
        problems = self.completeness_problems(min_observations)
        if problems:
            self.state = INCOMPLETE
            self.end_time = end_time
            self.metadata["incomplete_reasons"] = "; ".join(problems)
            return self
        self.state = COMPLETED
        self.end_time = end_time
        return self

    def mark_incomplete(self, end_time: Optional[str] = None) -> "Capture":
        """ACTIVE -> INCOMPLETE explicitly (operator override)."""
        _check_transition(self.state, INCOMPLETE)
        self.state = INCOMPLETE
        self.end_time = end_time
        return self

    def abort(self, end_time: Optional[str] = None,
              reason: Optional[str] = None) -> "Capture":
        """{PLANNED, ACTIVE} -> ABORTED; reason stored if supplied."""
        _check_transition(self.state, ABORTED)
        self.state = ABORTED
        self.end_time = end_time
        if reason:
            self.metadata["abort_reason"] = reason
        return self

    # ------------------------------------------------------
    # Data attachment (raw preservation, §9)
    # ------------------------------------------------------

    def add_records(self, records: Sequence[MeasurementRecord]) -> None:
        """Attach RAW records. Only an ACTIVE capture accepts data."""
        if self.state != ACTIVE:
            raise CaptureNotActiveError(
                f"records can only be added to an ACTIVE capture "
                f"(current state: {self.state})"
            )
        self.records.extend(records)

    def attach_ground_truth(
        self,
        values: Dict[Tuple[str, int, Optional[int]], float],
        source: Optional[str] = None,
        uncertainty_m: Optional[float] = None,
        note: Optional[str] = None,
    ) -> int:
        """
        Operator-supplied ground truth for this capture's records.

        Only ACTIVE captures accept truth (audit ordering); returns the
        number of records that received a value. Missing keys keep
        ground_truth=None. The attachment is AUDITED (§11); records are
        never mutated — the truth is applied in derived_records().
        """
        if self.state != ACTIVE:
            raise CaptureNotActiveError(
                f"ground truth can only be attached to an ACTIVE capture "
                f"(current state: {self.state})"
            )
        if uncertainty_m is not None and uncertainty_m < 0:
            raise ValueError(
                f"uncertainty_m must be >= 0 (operator-supplied), got "
                f"{uncertainty_m}"
            )
        joined = 0
        for rec in self.records:
            key = rec.identity_key()
            if key in values:
                if values[key] <= 0:
                    raise ValueError(
                        "ground truth must be a positive physical distance, "
                        f"got {values[key]!r}"
                    )
                self.ground_truth_log.append(GroundTruthEntry(
                    scene_id=key[0],
                    frame_id=key[1],
                    track_id=key[2],
                    ground_truth_distance_m=float(values[key]),
                    source=source,
                    uncertainty_m=uncertainty_m,
                    note=note,
                ))
                joined += 1
        return joined

    # ------------------------------------------------------
    # Derived views (raw is NEVER mutated, §9)
    # ------------------------------------------------------

    def derived_records(self) -> List[MeasurementRecord]:
        """
        RAW records with audited ground truth applied — a DERIVED view
        regenerated on every call. The raw list is untouched.
        """
        truth = {}
        for e in self.ground_truth_log:
            truth.setdefault((e.scene_id, e.frame_id, e.track_id),
                             e.ground_truth_distance_m)
        out: List[MeasurementRecord] = []
        for rec in self.records:
            key = rec.identity_key()
            if key in truth and rec.ground_truth_distance_m is None:
                rec = replace(rec, ground_truth_distance_m=truth[key])
            out.append(rec)
        return out

    # ------------------------------------------------------
    # Completeness (§8 - documented rule)
    # ------------------------------------------------------

    def completeness_problems(self, min_observations: int = 1) -> List[str]:
        """
        The documented completeness rule, evaluated honestly. Ground
        truth is deliberately NOT part of capture completion (truth is
        a separate workflow step; its absence surfaces in the
        experiment completion report).
        """
        problems: List[str] = []
        if len(self.records) < min_observations:
            problems.append(
                f"insufficient observations: {len(self.records)} < "
                f"{min_observations}"
            )
        for i, rec in enumerate(self.records):
            if not rec.scene_id:
                problems.append(f"record {i}: missing scene identity")
            if rec.frame_id < 0:
                problems.append(f"record {i}: negative frame id")
            if rec.distance_provenance not in VALID_PROVENANCES:
                problems.append(
                    f"record {i}: provenance "
                    f"{rec.distance_provenance!r} outside project vocabulary"
                )
            if (not math.isfinite(rec.timestamp)) or rec.timestamp < 0:
                problems.append(f"record {i}: invalid timestamp")
        return problems

    # ------------------------------------------------------
    # Truth reporting
    # ------------------------------------------------------

    def missing_truth(self) -> int:
        """Records still lacking ground truth (from the DERIVED view)."""
        return sum(
            1 for r in self.derived_records()
            if r.ground_truth_distance_m is None
        )

    # ------------------------------------------------------
    # Serialization
    # ------------------------------------------------------

    def key(self) -> Tuple[float, int]:
        """The slot identity: (target_distance_m, repetition)."""
        return (self.target_distance_m, self.repetition)

    def to_dict(self) -> Dict[str, object]:
        return {
            "target_distance_m": self.target_distance_m,
            "repetition": self.repetition,
            "scene_id": self.scene_id,
            "state": self.state,
            "n_records": len(self.records),
            "n_truth_attached": len(self.ground_truth_log),
            "missing_truth": self.missing_truth(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "notes": self.notes,
            "metadata": dict(sorted(self.metadata.items())),
            "session": self.session.to_dict() if self.session else None,
            "ground_truth_audit": [e.to_dict() for e in self.ground_truth_log],
        }
