"""
calibration/protocol.py

CONTROLLED EXPERIMENT PROTOCOL (Phase 10 data-collection protocol).

Drives a full experiment over a TargetDistancePlan:

    ExperimentProtocol(plan, scenes, experiment_id)
        -> manifest()                 machine-readable audit (§14)
        -> start_capture(...)         open a slot (PLANNED -> ACTIVE)
        -> capture.add_records(...)   RAW data (never mutated, §9)
        -> capture.attach_ground_truth(...)  audited (§11)
        -> capture.complete()/abort() deterministic state machine (§5)
        -> assignment()               CALIBRATION/VALIDATION split (§13)
        -> assert_disjoint()          overlap fails loudly (§13)
        -> freeze()                   DATASET FROZEN (§17)
        -> completion()               honest completion report (§15)
        -> to_dict()/from_dict()      reproducible manifest (§19)

Honesty rules enforced here:

- REPEATED slots are never overwritten: resuming an ABORTED slot
  creates capture attempt #2 — the aborted attempt remains in the
  manifest exactly as it ended (§7/§16).
- DATASET FREEZE (§17): after freeze(), adding captures, starting
  captures, attaching data, or re-assigning datasets raises
  ProtocolFrozenError. Raw data on disk/datasets is immutable from the
  experiment layer.
- CALIBRATION/VALIDATION ASSIGNMENT (§13): explicit, at capture
  granularity. A capture assigned to one role can never be assigned to
  the other; re-assignment is rejected (immutability of the split).
  Disjointness of the resulting record sets is verified with
  metrics.assert_disjoint before an experiment runs.
- COMPLETION (§15): the report counts planned/completed/incomplete/
  aborted/missing-truth/missing-observations honestly — the protocol
  NEVER reports itself fully complete while planned slots are missing
  or captures are INCOMPLETE/ABORTED without an approved attempt
  covering them.
- TARGET vs GROUND TRUTH (§12): slots are keyed by the plan's target
  distance; operator ground truth lives only on records/audit entries.
- No fabricated metadata: device/version/environment fields stay None
  unless an operator supplies them (ExperimentSession handles that).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from evaluation.metrics import assert_disjoint
from evaluation.record import MeasurementRecord

from calibration.capture import (
    ABORTED,
    ACTIVE,
    COMPLETED,
    INCOMPLETE,
    PLANNED,
    Capture,
    GroundTruthEntry,
)
from calibration.plan import TargetDistancePlan
from calibration.session import ExperimentSession


class ProtocolFrozenError(RuntimeError):
    """Raised when the frozen experiment layer is asked to mutate."""


class ProtocolStateError(ValueError):
    """Raised on invalid protocol-level operations."""


# Dataset roles (§13)
CALIBRATION = "CALIBRATION"
VALIDATION = "VALIDATION"
UNASSIGNED = "UNASSIGNED"


@dataclass
class ExperimentProtocol:
    """
    Controlled capture protocol over a TargetDistancePlan.

    scenes: scene identifiers to capture (e.g. ("bench_a", "bench_b")).
    Planned SLOTS are derived from the plan (scene x target x
    repetition); `captures` is the append-only log of STARTED capture
    attempts (never placeholders). A slot with no attempt is PLANNED.
    """

    plan: TargetDistancePlan
    scenes: Tuple[str, ...]
    experiment_id: str = "experiment"
    software_version: Optional[str] = None   # operator-supplied only (§19)
    notes: Optional[str] = None

    captures: List[Capture] = field(default_factory=list, init=False)
    assignments: Dict[Tuple[str, float, int], str] = field(
        default_factory=dict, init=False
    )
    _frozen: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.scenes:
            raise ProtocolStateError(
                "scenes must contain at least one scene identifier"
            )
        if any(not s for s in self.scenes):
            raise ProtocolStateError("scene identifiers must be non-empty")
        # `captures` intentionally starts EMPTY: slots are virtual until
        # an operator starts them; PLANNED is a derived state.

    # ------------------------------------------------------
    # Introspection
    # ------------------------------------------------------

    def slot_key(self, capture: Capture) -> Tuple[str, float, int]:
        return (capture.scene_id or "", capture.target_distance_m,
                capture.repetition)

    def planned_slot_keys(self) -> List[Tuple[str, float, int]]:
        """
        Every planned (scene, target, repetition) slot in deterministic
        order (scene -> target -> repetition). Slots are virtual: a slot
        with no attempt is PLANNED.
        """
        return [
            (scene, float(target), rep)
            for scene in self.scenes
            for target in sorted(self.plan.targets_m)
            for rep in range(1, self.plan.repetitions + 1)
        ]

    def find_capture(
        self, scene_id: str, target_distance_m: float, repetition: int
    ) -> Optional[Capture]:
        """Latest capture attempt for a slot (resume support, §16)."""
        found = [
            c for c in self.captures
            if (c.scene_id == scene_id
                and c.target_distance_m == target_distance_m
                and c.repetition == repetition)
        ]
        return found[-1] if found else None

    def attempts(
        self, scene_id: str, target_distance_m: float, repetition: int
    ) -> List[Capture]:
        """ALL capture attempts for a slot, in creation order (§16)."""
        return [
            c for c in self.captures
            if (c.scene_id == scene_id
                and c.target_distance_m == target_distance_m
                and c.repetition == repetition)
        ]

    # ------------------------------------------------------
    # Capture lifecycle (guarded by freeze, §17)
    # ------------------------------------------------------

    def _require_unfrozen(self, action: str) -> None:
        if self._frozen:
            raise ProtocolFrozenError(
                f"dataset is FROZEN: {action} is no longer allowed from "
                "the experiment layer (§17)"
            )

    def start_capture(
        self,
        scene_id: str,
        target_distance_m: float,
        repetition: int,
        session: Optional[ExperimentSession] = None,
        start_time: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Capture:
        """
        Open the latest attempt for a slot (PLANNED -> ACTIVE).

        Resume semantics (§16): if the latest attempt is ABORTED (or
        terminal for any reason), a NEW attempt with the same
        (scene, target, repetition) is created and started — the
        earlier attempt is preserved untouched. Starting an already
        ACTIVE or terminal attempt is an error; the caller chooses to
        resume explicitly by calling start_capture again.
        """
        self._require_unfrozen("starting a capture")
        if target_distance_m not in self.plan.targets_m:
            raise ProtocolStateError(
                f"target {target_distance_m} m is not in the experiment "
                f"plan {sorted(self.plan.targets_m)}"
            )
        if not (1 <= repetition <= self.plan.repetitions):
            raise ProtocolStateError(
                f"repetition {repetition} outside plan range "
                f"1..{self.plan.repetitions}"
            )
        if scene_id not in self.scenes:
            raise ProtocolStateError(
                f"scene {scene_id!r} is not in the protocol scenes "
                f"{list(self.scenes)}"
            )
        current = self.find_capture(scene_id, target_distance_m, repetition)
        if current is not None and current.state == ACTIVE:
            raise ProtocolStateError(
                f"slot ({scene_id}, {target_distance_m}, {repetition}) is "
                "already ACTIVE"
            )
        if current is not None and current.state in (COMPLETED, INCOMPLETE):
            raise ProtocolStateError(
                f"slot ({scene_id}, {target_distance_m}, {repetition}) "
                f"already finished as {current.state}; resume is only "
                "possible after ABORT (§16)"
            )
        attempt_no = len(self.attempts(scene_id, target_distance_m,
                                       repetition)) + 1
        capture = Capture(
            target_distance_m=float(target_distance_m),
            repetition=repetition,
            scene_id=scene_id,
            session=session,
            notes=notes,
        )
        capture.metadata["experiment_id"] = self.experiment_id
        capture.metadata["attempt"] = str(attempt_no)
        capture.start(start_time=start_time)
        self.captures.append(capture)
        return capture

    def abort_capture(
        self,
        capture: Capture,
        end_time: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Abort one capture (deterministic state machine, §5)."""
        self._require_unfrozen("aborting a capture")
        capture.abort(end_time=end_time, reason=reason)

    # ------------------------------------------------------
    # Ground truth passthrough (audit lives on the capture, §11)
    # ------------------------------------------------------

    def attach_ground_truth(
        self,
        capture: Capture,
        values: Dict[Tuple[str, int, Optional[int]], float],
        source: Optional[str] = None,
        uncertainty_m: Optional[float] = None,
        note: Optional[str] = None,
    ) -> int:
        """Audited ground-truth attachment via the capture (§10/§11)."""
        self._require_unfrozen("attaching ground truth")
        return capture.attach_ground_truth(
            values, source=source, uncertainty_m=uncertainty_m, note=note
        )

    # ------------------------------------------------------
    # Dataset assignment (§13)
    # ------------------------------------------------------

    def assign(
        self,
        capture: Capture,
        role: str,
    ) -> None:
        """
        Assign a capture to CALIBRATION or VALIDATION.

        Re-assignment to the OTHER role is rejected (the split is
        immutable); re-assigning the same role is accepted (idempotent).
        Assignment requires the capture to be terminal — an active
        capture's final data extent is not yet known.
        """
        self._require_unfrozen("changing dataset assignment")
        if role not in (CALIBRATION, VALIDATION):
            raise ProtocolStateError(
                f"role must be {CALIBRATION} or {VALIDATION}, got {role!r}"
            )
        if capture.state not in (COMPLETED, INCOMPLETE):
            raise ProtocolStateError(
                f"only terminal captures can be assigned (current state: "
                f"{capture.state})"
            )
        key = self.slot_key(capture)
        existing = self.assignments.get(key)
        if existing is not None and existing != role:
            raise ProtocolStateError(
                f"slot {key} is already assigned to {existing}; the "
                "calibration/validation split is immutable (§13)"
            )
        self.assignments[key] = role

    def assignment(
        self, scene_id: str, target_distance_m: float, repetition: int
    ) -> str:
        """Role of a slot, or UNASSIGNED."""
        return self.assignments.get(
            (scene_id, target_distance_m, repetition), UNASSIGNED
        )

    def dataset_records(
        self, role: str, include_unassigned: bool = False
    ) -> List[MeasurementRecord]:
        """
        DERIVED records (ground truth applied) of all captures assigned
        to `role`. Only the LATEST attempt per slot feeds the dataset —
        superseded attempts (e.g. an aborted attempt before a resume)
        are preserved in the manifest but never contaminate the split
        (§9/§16). Raw records are never returned modified — truth
        application happens in the derived view (§9).
        """
        if role not in (CALIBRATION, VALIDATION):
            raise ProtocolStateError(
                f"role must be {CALIBRATION} or {VALIDATION}, got {role!r}"
            )
        out: List[MeasurementRecord] = []
        for key in self.planned_slot_keys():
            slot_role = self.assignments.get(key)
            if slot_role == role or (include_unassigned and slot_role is None):
                latest = self.find_capture(*key)
                if (latest is not None
                        and latest.state in (COMPLETED, INCOMPLETE)):
                    out.extend(latest.derived_records())
        return out

    def verify_split(self) -> None:
        """
        Loud disjointness verification of the calibration/validation
        record sets (§13) — reuses evaluation.metrics.assert_disjoint.
        """
        assert_disjoint(
            self.dataset_records(CALIBRATION),
            self.dataset_records(VALIDATION),
        )

    # ------------------------------------------------------
    # Freeze (§17)
    # ------------------------------------------------------

    def freeze(self) -> None:
        """
        Freeze the dataset: captures, assignments, and ground truth
        become immutable from the experiment layer. Idempotent.
        """
        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    # ------------------------------------------------------
    # Completion report (§15)
    # ------------------------------------------------------

    def completion(self) -> Dict[str, object]:
        """
        Honest completion report over PLANNED SLOTS (one slot per
        (scene, target, repetition)). A slot with no attempt is
        PLANNED; otherwise its LATEST attempt decides the state. A slot
        counts as complete only if its latest attempt is COMPLETED —
        ABORTED/INCOMPLETE attempts are reported, never silently
        converted into successes (§16).
        """
        latest: Dict[Tuple[str, float, int], Capture] = {}
        for capture in self.captures:
            latest[self.slot_key(capture)] = capture  # creation order

        planned_slots = self.planned_slot_keys()
        planned = len(planned_slots)
        states = {PLANNED: 0, ACTIVE: 0, COMPLETED: 0,
                  INCOMPLETE: 0, ABORTED: 0}
        missing_truth = 0
        n_records_captured = 0
        attempts_per_slot: List[Dict[str, object]] = []

        for key in sorted(latest):
            capture = latest[key]
            states[capture.state] = states.get(capture.state, 0) + 1
            if capture.state in (COMPLETED, INCOMPLETE):
                missing_truth += capture.missing_truth()
                n_records_captured += len(capture.records)
            attempts = self.attempts(*key)
            if len(attempts) > 1:
                attempts_per_slot.append({
                    "scene_id": key[0],
                    "target_distance_m": key[1],
                    "repetition": key[2],
                    "attempts": len(attempts),
                    "states": [a.state for a in attempts],
                })
        for key in planned_slots:
            if key not in latest:
                states[PLANNED] += 1

        # Missing observations = planned records (one repetition's
        # worth per slot) minus records in terminal latest attempts —
        # an exact count; ACTIVE attempts count only what they hold
        # once they terminate.
        planned_records = planned * self.plan.repetitions
        missing_observations = max(0, planned_records - n_records_captured)
        fully_complete = (
            states[COMPLETED] == planned
            and missing_truth == 0
        )
        report: Dict[str, object] = {
            "experiment_id": self.experiment_id,
            "planned_slots": planned,
            "slots": dict(sorted(states.items())),
            "captures_total": len(self.captures),
            "extra_attempts": len(self.captures) - len(latest),
            "missing_ground_truth_records": missing_truth,
            "missing_observations": missing_observations,
            "missing_truth_slots": [
                list(k) for k in sorted(latest)
                if latest[k].state in (COMPLETED, INCOMPLETE)
                and latest[k].missing_truth() > 0
            ],
            "resumed_slots": attempts_per_slot,
            "fully_complete": fully_complete,
            "frozen": self._frozen,
            "assignment_counts": {
                CALIBRATION: sum(
                    1 for v in self.assignments.values() if v == CALIBRATION),
                VALIDATION: sum(
                    1 for v in self.assignments.values() if v == VALIDATION),
                UNASSIGNED: planned - len(self.assignments),
            },
        }
        return report

    # ------------------------------------------------------
    # Manifest (§14) - machine-readable, deterministic
    # ------------------------------------------------------

    def manifest(self) -> Dict[str, object]:
        """Full, reproducible experiment manifest (§14/§19)."""
        return {
            "schema": "sina-capture-manifest/1",
            "experiment_id": self.experiment_id,
            "frozen": self._frozen,
            "software_version": self.software_version,  # None if unknown
            "notes": self.notes,
            "plan": self.plan.summary(),
            "scenes": list(self.scenes),
            "planned_slots": len(self.planned_slot_keys()),
            "assignment_counts": {
                CALIBRATION: sum(
                    1 for v in self.assignments.values() if v == CALIBRATION),
                VALIDATION: sum(
                    1 for v in self.assignments.values() if v == VALIDATION),
            },
            "captures": [
                c.to_dict() for c in self.captures if c.state != PLANNED
            ],
            "assignments": {
                f"{k[0]}|{k[1]}|{k[2]}": role
                for k, role in sorted(self.assignments.items())
            },
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.manifest(), sort_keys=True, indent=2)

    @classmethod
    def from_manifest(
        cls, data: Dict[str, object]
    ) -> "ExperimentProtocol":
        """
        Rehydrate a protocol from its manifest for AUDIT/inspection.

        The rehydrated protocol is FROZEN: manifests are records of
        what happened, not live mutable state — reconstructing must
        never let new captures silently alter a recorded experiment
        (§17/§19). Capture records/audit are not serialized in the
        manifest (records live in the JSONL datasets); the rehydrated
        captures carry their lifecycle state and metadata only.
        """
        import json
        plan_data = data["plan"]
        plan = TargetDistancePlan(
            targets_m=tuple(plan_data["targets_m"]),  # type: ignore[arg-type]
            repetitions=int(plan_data["repetitions_per_target"]),  # type: ignore[arg-type]
        )
        protocol = cls(
            plan=plan,
            scenes=tuple(data["scenes"]),  # type: ignore[arg-type]
            experiment_id=str(data["experiment_id"]),
            software_version=data.get("software_version"),  # type: ignore[arg-type]
            notes=data.get("notes"),  # type: ignore[arg-type]
        )
        protocol.captures = []
        for cd in data["captures"]:  # type: ignore[union-attr]
            capture = Capture(
                target_distance_m=float(cd["target_distance_m"]),
                repetition=int(cd["repetition"]),
                scene_id=cd.get("scene_id"),
                notes=cd.get("notes"),
            )
            capture.state = str(cd["state"])
            capture.start_time = cd.get("start_time")
            capture.end_time = cd.get("end_time")
            capture.metadata = {
                str(k): str(v)
                for k, v in (cd.get("metadata") or {}).items()
            }
            # Ground-truth AUDIT is serialized in the manifest and must
            # survive rehydration (§17: ground-truth metadata preserved).
            capture.ground_truth_log = [
                GroundTruthEntry(
                    scene_id=str(e["scene_id"]),
                    frame_id=int(e["frame_id"]),
                    track_id=(
                        None if e.get("track_id") is None
                        else int(e["track_id"])
                    ),
                    ground_truth_distance_m=float(
                        e["ground_truth_distance_m"]
                    ),
                    source=e.get("source"),
                    uncertainty_m=e.get("uncertainty_m"),
                    note=e.get("note"),
                )
                for e in (cd.get("ground_truth_audit") or [])
            ]
            # Persisted derived counts (record payloads are not in the
            # manifest): the audit copy reports what each capture held.
            capture.record_count_audit = int(cd.get("n_records") or 0)
            capture.missing_truth_audit = int(cd.get("missing_truth") or 0)
            protocol.captures.append(capture)
        protocol.assignments = {
            (parts[0], float(parts[1]), int(parts[2])): str(role)
            for parts, role in (
                (str(k).split("|"), v)
                for k, v in (data.get("assignments") or {}).items()
            )
        }
        protocol._frozen = True
        return protocol
