"""
tests/test_capture_protocol.py

PHASE 10 DATA-COLLECTION PROTOCOL TESTS. HARDWARE-FREE. All fixtures
SIMULATED (provenance "SIMULATED", sim_* scenes) - nothing here claims
MEASURED OAK-D data or real-world accuracy.

Covers (milestone §21):

  - session creation
  - state transitions and invalid transitions
  - target plan integration
  - repetition tracking (no overwriting, §7)
  - capture completeness (§8)
  - raw-data preservation (§9)
  - ground-truth attachment + audit (§10/§11)
  - missing ground truth
  - target vs ground-truth separation (§12)
  - calibration/validation assignment (§13)
  - overlap rejection (§13)
  - manifest serialization (§14)
  - completion report (§15)
  - abort / resume (§16)
  - dataset freeze + mutation rejection (§17)
  - frozen calibration preserved (§18)
  - deterministic output
  - simulated provenance
"""

import json
from pathlib import Path

import pytest

from evaluation.record import MeasurementRecord
from evaluation.metrics import assert_disjoint

from calibration import (
    ABORTED,
    ACTIVE,
    AffineCalibration,
    CALIBRATION,
    CalibrationExperiment,
    Capture,
    CaptureNotActiveError,
    CaptureStateError,
    COMPLETED,
    ExperimentProtocol,
    ExperimentSession,
    INCOMPLETE,
    IdentityCalibration,
    PLANNED,
    ProtocolFrozenError,
    ProtocolStateError,
    TargetDistancePlan,
    UNASSIGNED,
    VALIDATION,
)


# ==============================================================
# Fixtures - ALL SIMULATED
# ==============================================================

def mkrec(
    scene: str = "sim_bench",
    frame: int = 0,
    label: str = "person",
    pred=None,
    gt=None,
    prov: str = "SIMULATED",
    track: int = 0,
    timestamp: float = 0.0,
) -> MeasurementRecord:
    return MeasurementRecord(
        schema_version=1,
        scene_id=scene,
        frame_id=frame,
        timestamp=timestamp,
        track_id=track,
        label=label,
        confidence=0.9,
        bbox=(100, 100, 200, 300),
        region="CENTER",
        predicted_distance_m=pred,
        distance_provenance=prov,
        distance_source="mock_bbox",
        ground_truth_distance_m=gt,
    )


def fill_capture(capture: Capture, target: float, rep: int,
                 bias: float = 1.1, frames=None) -> None:
    """Start + populate + complete one capture with SIMULATED records."""
    capture.start(start_time="2026-09-14T10:00:00")
    frames = frames if frames is not None else [rep * 10]
    recs = [
        mkrec(scene=capture.scene_id or "sim_bench", frame=f,
              pred=round(bias * target, 4),
              timestamp=1000.0 + f)
        for f in frames
    ]
    capture.add_records(recs)
    capture.complete(end_time="2026-09-14T10:00:05")


def small_protocol() -> ExperimentProtocol:
    return ExperimentProtocol(
        plan=TargetDistancePlan(targets_m=(1.0, 2.0), repetitions=2),
        scenes=("sim_a", "sim_b"),
        experiment_id="sim_exp_001",
        software_version="f842deb",
    )


# ==============================================================
# Capture state machine (§5)
# ==============================================================

class TestCaptureStateMachine:
    def test_full_happy_path(self):
        c = Capture(target_distance_m=2.0, repetition=1, scene_id="sim_a")
        assert c.state == PLANNED
        c.start()
        assert c.state == ACTIVE
        c.add_records([mkrec(pred=2.0)])
        c.complete()
        assert c.state == COMPLETED

    def test_invalid_transitions_rejected(self):
        c = Capture(target_distance_m=2.0, repetition=1)
        with pytest.raises(CaptureStateError, match="invalid transition"):
            c.complete()               # PLANNED -> COMPLETED not allowed
        c.abort()
        with pytest.raises(CaptureStateError, match="no-op"):
            c.abort()                  # terminal -> terminal not allowed

    def test_terminal_states_are_terminal(self):
        c = Capture(target_distance_m=2.0, repetition=1)
        c.start(); c.complete()
        for action in (c.start, c.complete, c.mark_incomplete, c.abort):
            with pytest.raises(CaptureStateError):
                action()
        a = Capture(target_distance_m=2.0, repetition=1)
        a.start(); a.abort()
        with pytest.raises(CaptureStateError):
            a.start()                  # resume happens at protocol level

    def test_no_op_transitions_rejected(self):
        c = Capture(target_distance_m=2.0, repetition=1)
        c.start()
        with pytest.raises(CaptureStateError, match="no-op"):
            c.start()

    def test_records_require_active(self):
        c = Capture(target_distance_m=2.0, repetition=1)
        with pytest.raises(CaptureNotActiveError):
            c.add_records([mkrec(pred=2.0)])
        c.start(); c.complete()
        with pytest.raises(CaptureNotActiveError):
            c.add_records([mkrec(pred=2.0)])

    def test_interrupted_is_never_completed(self):
        c = Capture(target_distance_m=2.0, repetition=1)
        c.start()
        c.abort(reason="USB hiccup")   # operator-interrupted
        assert c.state == ABORTED
        assert c.metadata["abort_reason"] == "USB hiccup"


# ==============================================================
# Completeness (§8)
# ==============================================================

class TestCompleteness:
    def test_complete_with_valid_records(self):
        c = Capture(target_distance_m=2.0, repetition=1, scene_id="sim_a")
        c.start()
        c.add_records([mkrec(scene="sim_a", pred=2.0, timestamp=1.0)])
        c.complete()
        assert c.state == COMPLETED

    def test_no_observations_becomes_incomplete_not_discarded(self):
        c = Capture(target_distance_m=2.0, repetition=1)
        c.start()
        c.complete()
        assert c.state == INCOMPLETE
        assert "insufficient observations" in c.metadata["incomplete_reasons"]

    def test_bad_provenance_marks_incomplete(self):
        c = Capture(target_distance_m=2.0, repetition=1)
        c.start()
        c.add_records([mkrec(pred=2.0, prov="MADE_UP")])
        c.complete()
        assert c.state == INCOMPLETE

    def test_min_observations_respected(self):
        c = Capture(target_distance_m=2.0, repetition=1)
        c.start()
        c.add_records([mkrec(pred=2.0)])
        c.complete(min_observations=5)
        assert c.state == INCOMPLETE

    def test_explicit_mark_incomplete(self):
        c = Capture(target_distance_m=2.0, repetition=1)
        c.start()
        c.mark_incomplete()
        assert c.state == INCOMPLETE


# ==============================================================
# Repetition tracking (§7) and plan integration (§6)
# ==============================================================

class TestRepetitionsAndPlan:
    def test_slots_match_plan(self):
        protocol = small_protocol()
        # 2 scenes x 2 targets x 2 reps = 8 planned (virtual) slots;
        # captures start EMPTY - a slot with no attempt is PLANNED.
        assert len(protocol.planned_slot_keys()) == 8
        assert protocol.captures == []
        assert protocol.completion()["slots"]["PLANNED"] == 8
        assert protocol.find_capture("sim_a", 2.0, 1) is None
        assert protocol.find_capture("sim_a", 3.0, 1) is None

    def test_repetitions_remain_individually_recoverable(self):
        protocol = small_protocol()
        c1 = protocol.start_capture("sim_a", 2.0, 1)
        c1.add_records([mkrec(scene="sim_a", frame=1, pred=2.2)])
        c1.complete()
        c2 = protocol.start_capture("sim_a", 2.0, 2)
        c2.add_records([mkrec(scene="sim_a", frame=2, pred=2.3)])
        c2.complete()
        r1 = protocol.find_capture("sim_a", 2.0, 1).records
        r2 = protocol.find_capture("sim_a", 2.0, 2).records
        assert r1[0].frame_id == 1 and r2[0].frame_id == 2  # not overwritten

    def test_out_of_plan_targets_rejected(self):
        protocol = small_protocol()
        with pytest.raises(ProtocolStateError, match="not in the experiment plan"):
            protocol.start_capture("sim_a", 3.0, 1)
        with pytest.raises(ProtocolStateError):
            protocol.start_capture("sim_a", 2.0, 5)
        with pytest.raises(ProtocolStateError):
            protocol.start_capture("sim_c", 2.0, 1)

    def test_plan_targets_stay_experiment_targets(self):
        summary = TargetDistancePlan(targets_m=(1.0, 2.0), repetitions=2).summary()
        assert summary["kind"] == "EXPERIMENT TARGETS - not navigation/safety thresholds"


# ==============================================================
# Raw preservation (§9) and ground truth (§10-§12)
# ==============================================================

class TestRawAndGroundTruth:
    def test_raw_records_never_mutated_by_truth(self):
        c = Capture(target_distance_m=2.0, repetition=1, scene_id="sim_a")
        c.start()
        rec = mkrec(scene="sim_a", frame=1, pred=2.2, track=1)
        c.add_records([rec])
        c.attach_ground_truth({("sim_a", 1, 1): 1.97}, source="tape measure")
        derived = c.derived_records()
        assert derived[0].ground_truth_distance_m == 1.97  # derived view
        assert rec.ground_truth_distance_m is None          # raw untouched
        assert c.records[0].ground_truth_distance_m is None

    def test_truth_audit_entry_recorded(self):
        c = Capture(target_distance_m=2.0, repetition=1, scene_id="sim_a")
        c.start()
        c.add_records([mkrec(scene="sim_a", frame=1, pred=2.2, track=1)])
        n = c.attach_ground_truth(
            {("sim_a", 1, 1): 1.97}, source="tape measure",
            uncertainty_m=0.005, note="operator Norris",
        )
        assert n == 1
        entry = c.ground_truth_log[0]
        assert entry.ground_truth_distance_m == 1.97
        assert entry.source == "tape measure"
        assert entry.uncertainty_m == 0.005   # explicitly supplied only
        assert entry.note == "operator Norris"

    def test_uncertainty_never_invented(self):
        c = Capture(target_distance_m=2.0, repetition=1, scene_id="sim_a")
        c.start()
        c.add_records([mkrec(scene="sim_a", frame=1, pred=2.2, track=1)])
        c.attach_ground_truth({("sim_a", 1, 1): 1.97})
        assert c.ground_truth_log[0].uncertainty_m is None
        assert c.ground_truth_log[0].source is None

    def test_nonpositive_truth_rejected(self):
        c = Capture(target_distance_m=2.0, repetition=1, scene_id="sim_a")
        c.start()
        c.add_records([mkrec(scene="sim_a", frame=1, pred=2.2, track=1)])
        with pytest.raises(ValueError, match="positive physical distance"):
            c.attach_ground_truth({("sim_a", 1, 1): 0.0})

    def test_missing_truth_stays_none(self):
        c = Capture(target_distance_m=2.0, repetition=1, scene_id="sim_a")
        c.start()
        c.add_records([
            mkrec(scene="sim_a", frame=1, pred=2.2, track=1),
            mkrec(scene="sim_a", frame=2, pred=2.2, track=2),
        ])
        c.attach_ground_truth({("sim_a", 1, 1): 1.97})
        assert c.missing_truth() == 1
        derived = c.derived_records()
        assert derived[1].ground_truth_distance_m is None  # not zero

    def test_target_vs_ground_truth_separate(self):
        c = Capture(target_distance_m=2.0, repetition=1, scene_id="sim_a")
        c.start()
        c.add_records([mkrec(scene="sim_a", frame=1, pred=2.2, track=1)])
        c.attach_ground_truth({("sim_a", 1, 1): 1.97})
        assert c.target_distance_m == 2.0                    # plan value
        assert c.derived_records()[0].ground_truth_distance_m == 1.97
        assert c.target_distance_m != c.derived_records()[0].ground_truth_distance_m

    def test_truth_requires_active(self):
        c = Capture(target_distance_m=2.0, repetition=1)
        with pytest.raises(CaptureNotActiveError):
            c.attach_ground_truth({("s", 1, 1): 1.0})


# ==============================================================
# Dataset assignment + disjointness (§13)
# ==============================================================

class TestAssignment:
    def test_assign_terminal_only(self):
        protocol = small_protocol()
        c = protocol.start_capture("sim_a", 2.0, 1)
        with pytest.raises(ProtocolStateError, match="terminal"):
            protocol.assign(c, CALIBRATION)
        c.complete()
        protocol.assign(c, CALIBRATION)
        assert protocol.assignment("sim_a", 2.0, 1) == CALIBRATION

    def test_reassignment_to_other_role_rejected(self):
        protocol = small_protocol()
        c = protocol.start_capture("sim_a", 2.0, 1)
        c.complete()
        protocol.assign(c, CALIBRATION)
        with pytest.raises(ProtocolStateError, match="immutable"):
            protocol.assign(c, VALIDATION)

    def test_same_role_reassignment_idempotent(self):
        protocol = small_protocol()
        c = protocol.start_capture("sim_a", 2.0, 1)
        c.complete()
        protocol.assign(c, CALIBRATION)
        protocol.assign(c, CALIBRATION)  # accepted
        assert protocol.assignment("sim_a", 2.0, 1) == CALIBRATION

    def test_invalid_role_rejected(self):
        protocol = small_protocol()
        c = protocol.start_capture("sim_a", 2.0, 1)
        c.complete()
        with pytest.raises(ProtocolStateError, match="role must be"):
            protocol.assign(c, "SAFETY")

    def test_split_datasets_are_disjoint(self):
        protocol = small_protocol()
        for scene, role in (("sim_a", CALIBRATION), ("sim_b", VALIDATION)):
            for target in (1.0, 2.0):
                for rep in (1, 2):
                    c = protocol.start_capture(scene, target, rep)
                    c.add_records([
                        mkrec(scene=scene, frame=int(target * 10) + rep,
                              pred=target * 1.1, track=rep)
                    ])
                    c.attach_ground_truth(
                        {(scene, int(target * 10) + rep, rep): target}
                    )
                    c.complete()
                    protocol.assign(c, role)
        protocol.verify_split()  # must not raise
        calib = protocol.dataset_records(CALIBRATION)
        val = protocol.dataset_records(VALIDATION)
        assert calib and val
        assert all(r.distance_provenance == "SIMULATED" for r in calib + val)

    def test_overlap_fails_loudly(self):
        protocol = small_protocol()
        c = protocol.start_capture("sim_a", 2.0, 1)
        c.add_records([mkrec(scene="sim_a", frame=1, pred=2.2, track=1)])
        c.complete()
        protocol.assign(c, CALIBRATION)
        # Simulate a misassignment by placing the same identity into a
        # validation-role dataset directly (cross-check the guard).
        calib = protocol.dataset_records(CALIBRATION)
        val = [mkrec(scene="sim_a", frame=1, pred=2.2, track=1,
                     gt=None)]
        # give the validation copy the same identity key
        from dataclasses import replace
        val = [replace(val[0], ground_truth_distance_m=1.97)]
        with pytest.raises(ValueError, match="overlap detected"):
            assert_disjoint(calib, val)


# ==============================================================
# Abort / resume (§16)
# ==============================================================

class TestAbortResume:
    def test_aborted_attempt_preserved_and_resumed(self):
        protocol = small_protocol()
        c1 = protocol.start_capture("sim_a", 2.0, 1)
        c1.add_records([mkrec(scene="sim_a", frame=1, pred=2.2)])
        c1.abort(reason="device dropped")
        assert c1.state == ABORTED

        c2 = protocol.start_capture("sim_a", 2.0, 1)   # resume
        assert c2 is not c1
        assert c2.state == ACTIVE
        assert c2.records == []                        # fresh attempt
        c2.add_records([mkrec(scene="sim_a", frame=5, pred=2.2)])
        c2.complete()

        attempts = protocol.attempts("sim_a", 2.0, 1)
        assert [a.state for a in attempts] == [ABORTED, COMPLETED]
        # The abandoned PLANNED placeholder is gone: captures are the
        # started attempts only.
        assert len(attempts) == len([c for c in protocol.captures
                                     if c.state != PLANNED])
        # The aborted attempt keeps its partial record; nothing merged.
        assert len(attempts[0].records) == 1
        assert len(attempts[1].records) == 1
        # Latest attempt decides the slot; experiment uses the resume.
        assert protocol.find_capture("sim_a", 2.0, 1) is c2

    def test_cannot_start_over_completed(self):
        protocol = small_protocol()
        c = protocol.start_capture("sim_a", 2.0, 1)
        c.complete()
        with pytest.raises(ProtocolStateError, match="resume is only possible"):
            protocol.start_capture("sim_a", 2.0, 1)

    def test_cannot_start_already_active(self):
        protocol = small_protocol()
        protocol.start_capture("sim_a", 2.0, 1)
        with pytest.raises(ProtocolStateError, match="already ACTIVE"):
            protocol.start_capture("sim_a", 2.0, 1)

    def test_resume_after_incomplete_rejected(self):
        protocol = small_protocol()
        c = protocol.start_capture("sim_a", 2.0, 1)
        c.complete()  # no records -> INCOMPLETE
        with pytest.raises(ProtocolStateError):
            protocol.start_capture("sim_a", 2.0, 1)


# ==============================================================
# Freeze (§17) + frozen calibration (§18)
# ==============================================================

class TestFreeze:
    def _filled(self):
        protocol = small_protocol()
        c = protocol.start_capture("sim_a", 2.0, 1)
        c.add_records([mkrec(scene="sim_a", frame=1, pred=2.2, track=1)])
        c.attach_ground_truth({("sim_a", 1, 1): 1.97})
        c.complete()
        return protocol, c

    def test_freeze_blocks_all_mutations(self):
        protocol, c = self._filled()
        protocol.assign(c, CALIBRATION)
        protocol.freeze()
        assert protocol.frozen
        with pytest.raises(ProtocolFrozenError):
            protocol.start_capture("sim_b", 2.0, 1)
        with pytest.raises(ProtocolFrozenError):
            protocol.abort_capture(c)
        with pytest.raises(ProtocolFrozenError):
            protocol.attach_ground_truth(c, {("sim_a", 1, 1): 1.98})
        with pytest.raises(ProtocolFrozenError):
            protocol.assign(c, VALIDATION)

    def test_freeze_is_idempotent(self):
        protocol, _ = self._filled()
        protocol.freeze()
        protocol.freeze()
        assert protocol.frozen

    def test_reads_still_work_after_freeze(self):
        protocol, c = self._filled()
        protocol.assign(c, CALIBRATION)
        protocol.freeze()
        assert protocol.dataset_records(CALIBRATION)
        assert protocol.completion()["planned_slots"] == 8
        assert protocol.manifest()["frozen"] is True

    def test_frozen_calibration_workflow_unchanged(self):
        """capture -> fit -> freeze -> validation keeps FrozenCalibrationModel
        semantics (§18): validation never refits."""
        protocol = small_protocol()
        for scene, role in (("sim_a", CALIBRATION), ("sim_b", VALIDATION)):
            for target in (1.0, 2.0):
                for rep in (1, 2):
                    c = protocol.start_capture(scene, target, rep)
                    c.add_records([
                        mkrec(scene=scene, frame=int(target * 10) + rep,
                              pred=round(target * 1.1, 4), track=rep)
                    ])
                    c.attach_ground_truth(
                        {(scene, int(target * 10) + rep, rep): target}
                    )
                    c.complete()
                    protocol.assign(c, role)
        protocol.verify_split()
        protocol.freeze()

        result = CalibrationExperiment(
            candidates=[IdentityCalibration(), AffineCalibration()]
        ).run(
            protocol.dataset_records(CALIBRATION),
            protocol.dataset_records(VALIDATION),
        )
        assert result.selected_model == "affine"
        assert result.validation_metrics.mae_m < result.baseline_validation.mae_m


# ==============================================================
# Completion report (§15)
# ==============================================================

class TestCompletion:
    def test_empty_protocol_is_not_complete(self):
        protocol = small_protocol()
        report = protocol.completion()
        assert report["planned_slots"] == 8
        assert report["slots"]["PLANNED"] == 8
        assert report["fully_complete"] is False
        assert report["missing_observations"] == 16  # 8 slots x 2 reps

    def test_partial_completion_reported_honestly(self):
        protocol = small_protocol()
        # Complete 2 of 8 slots.
        for target in (1.0, 2.0):
            c = protocol.start_capture("sim_a", target, 1)
            c.add_records([
                mkrec(scene="sim_a", frame=int(target * 10),
                      pred=target * 1.1, track=1)
            ])
            c.attach_ground_truth({("sim_a", int(target * 10), 1): target})
            c.complete()
        report = protocol.completion()
        assert report["slots"]["COMPLETED"] == 2
        assert report["slots"]["PLANNED"] == 6
        assert report["fully_complete"] is False
        assert report["missing_observations"] == 14  # 7 open slots x 2

    def test_completed_without_truth_counts_missing_truth(self):
        protocol = small_protocol()
        c = protocol.start_capture("sim_a", 2.0, 1)
        c.add_records([
            mkrec(scene="sim_a", frame=1, pred=2.2, track=1),
            mkrec(scene="sim_a", frame=2, pred=2.2, track=2),
        ])
        c.complete()
        report = protocol.completion()
        assert report["slots"]["COMPLETED"] == 1
        assert report["missing_ground_truth_records"] == 2
        assert report["fully_complete"] is False
        assert (("sim_a", 2.0, 1) in
                {tuple(k) for k in report["missing_truth_slots"]})

    def test_aborted_slot_not_silently_success(self):
        protocol = small_protocol()
        c = protocol.start_capture("sim_a", 2.0, 1)
        c.add_records([mkrec(scene="sim_a", frame=1, pred=2.2)])
        c.abort(reason="device dropped")
        report = protocol.completion()
        assert report["slots"]["ABORTED"] == 1
        assert report["fully_complete"] is False

    def test_resume_tracked_in_report(self):
        protocol = small_protocol()
        c1 = protocol.start_capture("sim_a", 2.0, 1)
        c1.abort(reason="x")
        protocol.start_capture("sim_a", 2.0, 1)
        report = protocol.completion()
        assert report["extra_attempts"] == 1
        assert report["resumed_slots"][0]["attempts"] == 2
        assert report["resumed_slots"][0]["states"] == [ABORTED, ACTIVE]


# ==============================================================
# Manifest (§14) + reproducibility (§19)
# ==============================================================

class TestManifest:
    def test_manifest_structure_and_determinism(self):
        protocol = small_protocol()
        c = protocol.start_capture("sim_a", 2.0, 1,
                                   session=ExperimentSession(
                                       session_id="s1",
                                       device_id=None))  # unknown stays None
        c.add_records([mkrec(scene="sim_a", frame=1, pred=2.2)])
        c.complete()
        protocol.assign(c, CALIBRATION)

        m1 = json.dumps(protocol.manifest(), sort_keys=True)
        m2 = json.dumps(protocol.manifest(), sort_keys=True)
        assert m1 == m2
        manifest = protocol.manifest()
        assert manifest["schema"] == "sina-capture-manifest/1"
        assert manifest["experiment_id"] == "sim_exp_001"
        assert manifest["software_version"] == "f842deb"
        assert manifest["plan"]["kind"].startswith("EXPERIMENT TARGETS")
        assert manifest["planned_slots"] == 8
        # Manifest lists STARTED attempts only (PLANNED slots are
        # derived from the plan, not serialized).
        assert len(manifest["captures"]) == 1
        assert manifest["captures"][0]["state"] == COMPLETED
        assert manifest["assignments"] == {"sim_a|2.0|1": CALIBRATION}

    def test_unknown_metadata_stays_none(self):
        protocol = small_protocol()
        c = protocol.start_capture("sim_a", 2.0, 1)
        c.abort()
        assert protocol.manifest()["software_version"] == "f842deb"
        cap = protocol.manifest()["captures"][0]
        assert cap["session"] is None  # no fabricated device info
        assert cap["state"] == ABORTED

    def test_manifest_round_trip_is_frozen_audit_copy(self):
        protocol = small_protocol()
        c = protocol.start_capture("sim_a", 2.0, 1)
        c.add_records([mkrec(scene="sim_a", frame=1, pred=2.2)])
        c.complete()
        protocol.assign(c, CALIBRATION)
        data = json.loads(protocol.to_json())
        revived = ExperimentProtocol.from_manifest(data)
        assert revived.frozen                     # audit copies cannot mutate
        assert revived.experiment_id == "sim_exp_001"
        assert len(revived.captures) == 1         # started attempts only
        assert revived.captures[0].state == COMPLETED
        assert revived.assignment("sim_a", 2.0, 1) == CALIBRATION
        with pytest.raises(ProtocolFrozenError):
            revived.start_capture("sim_a", 2.0, 1)
        # The live protocol is unaffected by rehydration.
        assert protocol.frozen is False

    def test_manifest_writes_to_disk(self, tmp_path: Path):
        protocol = small_protocol()
        (tmp_path / "manifest.json").write_text(protocol.to_json())
        loaded = json.loads((tmp_path / "manifest.json").read_text())
        assert loaded["experiment_id"] == "sim_exp_001"


# ==============================================================
# Source-level guard: protocol imports no navigation config
# ==============================================================

class TestSourceGuards:
    def test_new_modules_import_no_navigation(self):
        import ast
        import calibration.capture
        import calibration.protocol
        from pathlib import Path
        offenders = []
        for mod in (calibration.capture, calibration.protocol):
            path = Path(mod.__file__)
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.split(".")[0] == "navigation":
                        offenders.append(path.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] == "navigation":
                            offenders.append(path.name)
        assert offenders == []
