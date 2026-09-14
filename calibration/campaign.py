"""
calibration/campaign.py

SYNTHETIC END-TO-END CALIBRATION CAMPAIGN (Phase 11).

Runs the COMPLETE offline calibration workflow over the REAL Phase 10
capture protocol with a deterministic synthetic sensor:

    TargetDistancePlan
        -> ExperimentProtocol (virtual slots -> attempt history)
        -> deterministic synthetic captures (SIMULATED provenance)
        -> abort/resume + incomplete + untruthed scenarios (§5/§6/§7)
        -> protocol-level CALIBRATION/VALIDATION assignment (§4)
        -> verify_split()  (overlap fails loudly, §12)
        -> baseline metrics on BOTH sets (§9)
        -> candidate fit on CALIBRATION DATA ONLY (§10)
        -> FrozenCalibrationModel -> validation metrics (§11/§20)
        -> coverage acceptance + acceptance gates (§13/§15)
        -> dataset freeze (§18) -> deterministic report (§19)

THE KNOWN GENERATING MODEL (§3) — encoded explicitly, never hidden:

    predicted_distance_m = ground_truth_distance_m * SCALE + OFFSET
                           + deterministic_jitter(frame_id)

with SCALE=1.10, OFFSET=0.08 and a pure-arithmetic jitter of at most
+/-amplitude (NO random number generator anywhere — identical config
produces byte-identical campaigns). The campaign therefore KNOWS the
true bias and the acceptance gates can verify the fitted model
recovers it on held-out validation data.

HONESTY BOUNDARY (§20): everything here is SIMULATED (provenance
"SIMULATED", scenes prefixed "sim_"). Passing the gates proves the
SOFTWARE workflow, not physical OAK-D measurement accuracy. Phase 9
hardware remains BLOCKED; no real-world claim is made or supportable.
"""

import json
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from evaluation.metrics import evaluate
from evaluation.record import MeasurementRecord

from calibration.coverage import availability_breakdown, coverage_report
from calibration.diagnostics import outlier_report, repeatability_analysis
from calibration.experiment import (
    CalibrationExperiment,
    ExperimentResult,
    apply_calibration,
    evaluate_calibrated,
)
from calibration.models import (
    AffineCalibration,
    FrozenCalibrationModel,
    IdentityCalibration,
    compare_candidates,
)
from calibration.plan import TargetDistancePlan
from calibration.capture import COMPLETED, INCOMPLETE
from calibration.protocol import (
    CALIBRATION,
    VALIDATION,
    ExperimentProtocol,
)
from calibration.reporting import build_experiment_report, render_experiment_markdown


# --------------------------------------------------------------
# Deterministic "sensor" (§3)
# --------------------------------------------------------------

SYNTHETIC_SCALE = 1.10
SYNTHETIC_OFFSET_M = 0.08


def synthetic_jitter(frame_id: int, amplitude_m: float) -> float:
    """
    Deterministic pseudo-jitter in [-amplitude_m, +amplitude_m].

    Pure arithmetic on the frame id — no RNG, no time, no global state:
    the same frame id always yields the same jitter.
    """
    return amplitude_m * (((frame_id * 7) % 13) - 6) / 6.0


def scene_truth_delta_m(scene_id: str, scene_index: int) -> float:
    """
    Deterministic operator "placement error" between the plan's nominal
    target and the actual ground truth (§7): the target and the truth
    are DISTINCT quantities and this fixture keeps them distinct
    (+0.01 / +0.02 / +0.03 m by scene order). Non-zero by design.
    """
    return round(((scene_index % 3) + 1) * 0.01, 3)


# --------------------------------------------------------------
# Campaign configuration (fully explicit, fully deterministic)
# --------------------------------------------------------------

@dataclass(frozen=True)
class CampaignConfig:
    """Everything needed to reproduce the campaign byte-for-byte."""

    experiment_id: str = "phase11-synthetic-campaign"
    scenes: Tuple[str, ...] = ("sim_corridor", "sim_doorway", "sim_open_room")
    labels: Tuple[str, ...] = ("person", "chair", "table")
    targets_m: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
    repetitions: int = 2                 # slots per (scene, target)
    records_per_capture: int = 2

    # Known generating model (§3). validation_bias None = same model.
    calibration_bias: Tuple[float, float] = (SYNTHETIC_SCALE, SYNTHETIC_OFFSET_M)
    validation_bias: Optional[Tuple[float, float]] = None
    calibration_jitter_m: float = 0.004
    validation_jitter_m: Optional[float] = None   # None = same as calibration

    # Planned scenario slots (§5/§6/§7).
    abort_slot: Tuple[str, float, int] = ("sim_corridor", 2.0, 2)
    incomplete_slot: Tuple[str, float, int] = ("sim_doorway", 3.0, 2)
    untruthed_slot: Tuple[str, float, int] = ("sim_open_room", 4.0, 2)
    sentinel_prediction_m: float = 42.0   # marks superseded-attempt data
    outlier_prediction_m: float = 9.9     # surfaced by diagnostics, never deleted

    # Acceptance gates (§13/§15) - deterministic, documented.
    acceptance_mae_tolerance_m: float = 0.05
    coverage_match_tolerance_m: float = 0.05

    def to_dict(self) -> Dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "scenes": list(self.scenes),
            "labels": list(self.labels),
            "targets_m": list(self.targets_m),
            "repetitions_per_target": self.repetitions,
            "records_per_capture": self.records_per_capture,
            "generating_model": (
                "predicted = ground_truth * scale + offset + jitter(frame_id)"
            ),
            "calibration_bias": list(self.calibration_bias),
            "validation_bias": list(
                self.validation_bias
                if self.validation_bias is not None else self.calibration_bias
            ),
            "calibration_jitter_m": self.calibration_jitter_m,
            "validation_jitter_m": (
                self.validation_jitter_m
                if self.validation_jitter_m is not None
                else self.calibration_jitter_m
            ),
            "abort_slot": list(self.abort_slot),
            "incomplete_slot": list(self.incomplete_slot),
            "untruthed_slot": list(self.untruthed_slot),
            "acceptance_mae_tolerance_m": self.acceptance_mae_tolerance_m,
            "coverage_match_tolerance_m": self.coverage_match_tolerance_m,
        }


# --------------------------------------------------------------
# Acceptance gates (§13) - derived from MEASURED metrics only
# --------------------------------------------------------------

@dataclass(frozen=True)
class AcceptanceGate:
    """Deterministic gate verdicts; no 'calibration always passes' flag."""

    gates: Tuple[Tuple[str, bool, str], ...]

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.gates)

    def failed(self) -> Tuple[str, ...]:
        return tuple(name for name, ok, _ in self.gates if not ok)

    def to_dict(self) -> Dict[str, object]:
        return {
            "passed": self.passed,
            "gates": [
                {"gate": name, "passed": ok, "detail": detail}
                for name, ok, detail in self.gates
            ],
        }


def coverage_acceptance(
    calibration_records: Sequence[MeasurementRecord],
    validation_records: Sequence[MeasurementRecord],
    requested_targets_m: Sequence[float],
    tolerance_m: float = 0.05,
) -> Dict[str, object]:
    """
    EXPERIMENT TARGET coverage gate (§15): every requested target must
    appear (within tolerance) in BOTH splits' ground truths. This is an
    ANALYSIS acceptance gate, NOT a safety threshold. A campaign with a
    missing required target is not fully valid.
    """
    def _covered(records: Sequence[MeasurementRecord], target: float) -> bool:
        return any(
            r.ground_truth_distance_m is not None
            and abs(r.ground_truth_distance_m - target) <= tolerance_m
            for r in records
        )

    missing = {
        "calibration": [t for t in requested_targets_m
                        if not _covered(calibration_records, t)],
        "validation": [t for t in requested_targets_m
                       if not _covered(validation_records, t)],
    }
    return {
        "passed": not missing["calibration"] and not missing["validation"],
        "requested_targets_m": sorted(set(requested_targets_m)),
        "missing_targets_m": missing,
        "match_tolerance_m": tolerance_m,
        "note": (
            "EXPERIMENT TARGET coverage gate - analysis acceptance only, "
            "not a navigation/safety threshold."
        ),
    }


# --------------------------------------------------------------
# Campaign result
# --------------------------------------------------------------

@dataclass
class CampaignResult:
    """Everything the campaign produced; deterministic for a config."""

    config: CampaignConfig
    protocol: ExperimentProtocol
    calibration_records: List[MeasurementRecord]
    validation_records: List[MeasurementRecord]
    all_raw_records: Tuple[MeasurementRecord, ...]
    experiment: ExperimentResult
    frozen_model: FrozenCalibrationModel
    split_disjoint: bool
    coverage: Dict[str, object]
    acceptance: AcceptanceGate
    report: Dict[str, object]
    markdown: str

    @property
    def accepted(self) -> bool:
        return self.acceptance.passed


# --------------------------------------------------------------
# The campaign
# --------------------------------------------------------------

def _slot_role(rep: int) -> str:
    """Documented split rule: repetition 1 -> CALIBRATION, else VALIDATION."""
    return CALIBRATION if rep == 1 else VALIDATION


def _slot_bias(
    config: CampaignConfig, role: str
) -> Tuple[Tuple[float, float], float]:
    if role == CALIBRATION:
        return config.calibration_bias, config.calibration_jitter_m
    bias = (config.validation_bias if config.validation_bias is not None
            else config.calibration_bias)
    amp = (config.validation_jitter_m if config.validation_jitter_m is not None
           else config.calibration_jitter_m)
    return bias, amp


def _synthetic_records(
    config: CampaignConfig,
    scene_id: str,
    scene_index: int,
    target_m: float,
    rep: int,
    role: str,
) -> List[MeasurementRecord]:
    """Deterministic SIMULATED observations for one slot (§2/§3)."""
    truth_m = target_m + scene_truth_delta_m(scene_id, scene_index)
    (scale, offset), amp = _slot_bias(config, role)
    base_frame = int(round(target_m * 100)) + rep * 1000
    track_id = int(round(target_m * 10)) * 10 + rep
    label = config.labels[(scene_index + rep) % len(config.labels)]
    regions = ("LEFT", "CENTER", "RIGHT")

    records: List[MeasurementRecord] = []
    for i in range(config.records_per_capture):
        frame_id = base_frame + i
        predicted = truth_m * scale + offset + synthetic_jitter(frame_id, amp)
        records.append(MeasurementRecord(
            schema_version=1,
            scene_id=scene_id,
            frame_id=frame_id,
            timestamp=float(frame_id),          # synthetic clock, no wall time
            track_id=track_id,
            label=label,
            confidence=0.90,
            bbox=(10 * i + rep, 20, 10 * i + rep + 40, 120),
            region=regions[(i + rep) % len(regions)],
            predicted_distance_m=predicted,
            distance_provenance="SIMULATED",    # NEVER "MEASURED" (§22)
            distance_source="synthetic-bias-model",
            ground_truth_distance_m=None,       # truth attached later (§10)
        ))
    return records


def _truth_values(
    config: CampaignConfig,
    scene_id: str,
    scene_index: int,
    target_m: float,
    rep: int,
) -> Dict[Tuple[str, int, Optional[int]], float]:
    """Independent operator truth per record identity (§7/§10/§11)."""
    truth_m = target_m + scene_truth_delta_m(scene_id, scene_index)
    base_frame = int(round(target_m * 100)) + rep * 1000
    track_id = int(round(target_m * 10)) * 10 + rep
    return {
        (scene_id, base_frame + i, track_id): truth_m
        for i in range(config.records_per_capture)
    }


def run_campaign(config: Optional[CampaignConfig] = None) -> CampaignResult:
    """
    Execute the full deterministic synthetic campaign (§1/§28):

    VIRTUAL SLOT -> ATTEMPT HISTORY -> LATEST TERMINAL ATTEMPT ->
    DATASET MEMBERSHIP. Superseded attempts never enter a dataset.
    """
    config = config or CampaignConfig()
    plan = TargetDistancePlan(
        targets_m=config.targets_m, repetitions=config.repetitions
    )
    protocol = ExperimentProtocol(
        plan=plan,
        scenes=config.scenes,
        experiment_id=config.experiment_id,
        notes="Phase 11 synthetic campaign - SIMULATED data only",
    )
    scene_index = {s: i for i, s in enumerate(config.scenes)}

    # ---- 1. Planned slots -> captures (§2) -------------------------
    for slot in protocol.planned_slot_keys():
        scene_id, target_m, rep = slot
        role = _slot_role(rep)
        s_idx = scene_index[scene_id]

        if slot == config.abort_slot:
            # §5: attempt 1 captured then ABORTED (audit history);
            # attempt 2 resumes the same virtual slot and completes.
            # Attempt 1 uses DISTINCT frame ids (a real interruption
            # captures different frames) so superseded data can never
            # share an observation identity with the resumed attempt.
            attempt1 = protocol.start_capture(
                scene_id, target_m, rep, notes="planned abort scenario"
            )
            abort_base_frame = int(round(target_m * 100)) + rep * 1000 + 5000
            attempt1.add_records([
                MeasurementRecord(
                    schema_version=1,
                    scene_id=scene_id,
                    frame_id=abort_base_frame + i,
                    timestamp=float(abort_base_frame + i),
                    track_id=int(round(target_m * 10)) * 10 + rep,
                    label="person",
                    confidence=0.90,
                    bbox=(0, 0, 40, 120),
                    region="CENTER",
                    predicted_distance_m=config.sentinel_prediction_m,
                    distance_provenance="SIMULATED",
                    distance_source="synthetic-bias-model",
                    ground_truth_distance_m=None,
                )
                for i in range(config.records_per_capture)
            ])
            protocol.abort_capture(
                attempt1, reason="synthetic interruption (planned scenario)"
            )
            capture = protocol.start_capture(
                scene_id, target_m, rep, notes="resumed after abort"
            )
        else:
            capture = protocol.start_capture(scene_id, target_m, rep)

        # ---- 2. Raw records (SIMULATED provenance, §22) ------------
        records = _synthetic_records(config, scene_id, s_idx, target_m, rep, role)

        if slot == config.incomplete_slot:
            # §6: insufficient observations recorded; completion is
            # attempted and honestly yields INCOMPLETE (never discarded,
            # never completed, never assigned to a dataset). The single
            # kept observation is the deliberately injected OUTLIER
            # (§16) — surfaced by diagnostics, never deleted from raw.
            records = [replace(
                records[0],
                predicted_distance_m=config.outlier_prediction_m,
            )]
        capture.add_records(records)

        # ---- 3. Independent ground truth (§7) -----------------------
        if slot != config.untruthed_slot:
            protocol.attach_ground_truth(
                capture,
                _truth_values(config, scene_id, s_idx, target_m, rep),
                source="synthetic-operator",
                uncertainty_m=0.005,
                note="known synthetic placement",
            )

        # ---- 4. Lifecycle terminal state -----------------------------
        if slot == config.incomplete_slot:
            capture.complete(min_observations=config.records_per_capture)
            # INCOMPLETE captures are never assigned to a dataset (§6):
            # they stay visible in the completion report and manifest.
        else:
            capture.complete(min_observations=config.records_per_capture)
            protocol.assign(capture, role)

    # ---- 5. RAW snapshot BEFORE any derived view (§8/§11) ----------
    raw_snapshot = tuple(
        record for capture in protocol.captures for record in capture.records
    )

    # ---- 6. Split verification (§4/§12) ----------------------------
    protocol.verify_split()
    calibration_records = protocol.dataset_records(CALIBRATION)
    validation_records = protocol.dataset_records(VALIDATION)

    # ---- 7. Baseline FIRST (§9) ------------------------------------
    baseline_calibration = evaluate(calibration_records)
    baseline_validation = evaluate(validation_records)

    # ---- 8. Candidate fit on CALIBRATION DATA ONLY (§10/§19) -------
    comparison, frozen = compare_candidates(
        [IdentityCalibration(), AffineCalibration()], calibration_records
    )
    experiment = CalibrationExperiment(
        candidates=[IdentityCalibration(), AffineCalibration()]
    ).run(calibration_records, validation_records)

    # ---- 9. Coverage + diagnostics (§15/§16) ------------------------
    # Diagnostics describe ALL captured data: the assigned datasets AND
    # unassigned terminal captures (e.g. the INCOMPLETE slot with its
    # injected outlier) — an unassigned observation is still a real
    # captured observation. ACCURACY metrics, by contrast, use only the
    # assigned datasets (step 7/8). Raw records never carry truth.
    unassigned_derived: List[MeasurementRecord] = []
    for key in protocol.planned_slot_keys():
        if protocol.assignments.get(key) is None:
            latest = protocol.find_capture(*key)
            if (latest is not None
                    and latest.state in (COMPLETED, INCOMPLETE)):
                unassigned_derived.extend(latest.derived_records())
    derived_all = (
        list(calibration_records) + list(validation_records)
        + unassigned_derived
    )
    coverage = coverage_report(
        derived_all, requested_targets_m=config.targets_m
    )
    availability = availability_breakdown(derived_all)
    repeatability = [g.to_dict() for g in
                     repeatability_analysis(derived_all, min_samples=3)]
    outliers = outlier_report(derived_all, top_n=5)

    # ---- 10. Acceptance gates from MEASURED metrics (§13) ----------
    coverage_gate = coverage_acceptance(
        calibration_records, validation_records,
        config.targets_m, tolerance_m=config.coverage_match_tolerance_m,
    )
    val_cal_mae = experiment.validation_metrics.mae_m
    val_base_mae = experiment.baseline_validation.mae_m
    val_cal_rmse = experiment.validation_metrics.rmse_m
    val_base_rmse = experiment.baseline_validation.rmse_m
    gates: List[Tuple[str, bool, str]] = [
        (
            "split_disjoint",
            True,
            "protocol.verify_split() passed before any fitting",
        ),
        (
            "baseline_validation_mae_positive",
            (val_base_mae is not None and val_base_mae > 0),
            f"baseline validation MAE = {val_base_mae}",
        ),
        (
            "calibrated_validation_mae_improved",
            (
                val_cal_mae is not None and val_base_mae is not None
                and val_cal_mae < val_base_mae
            ),
            f"calibrated {val_cal_mae} < baseline {val_base_mae}",
        ),
        (
            "calibrated_validation_rmse_improved",
            (
                val_cal_rmse is not None and val_base_rmse is not None
                and val_cal_rmse < val_base_rmse
            ),
            f"calibrated {val_cal_rmse} < baseline {val_base_rmse}",
        ),
        (
            "calibrated_validation_mae_within_tolerance",
            (
                val_cal_mae is not None
                and val_cal_mae <= config.acceptance_mae_tolerance_m
            ),
            f"calibrated validation MAE {val_cal_mae} <= "
            f"{config.acceptance_mae_tolerance_m} m",
        ),
        (
            "coverage_complete",
            bool(coverage_gate["passed"]),
            f"missing targets: {coverage_gate['missing_targets_m']}",
        ),
    ]
    acceptance = AcceptanceGate(gates=tuple(gates))

    # ---- 11. Dataset freeze AFTER accepted data captured (§18) -----
    protocol.freeze()

    # ---- 12. Deterministic report (§19) -----------------------------
    experiment_report = build_experiment_report(
        experiment,
        calibration_records=calibration_records,
        validation_records=validation_records,
        requested_targets_m=config.targets_m,
    )
    campaign_report: Dict[str, object] = {
        "schema": "sina-synthetic-campaign/1",
        "phase": "11 (synthetic end-to-end campaign, software-only)",
        "hardware_validation": "BLOCKED - Phase 9 Checkpoint A not performed",
        "provenance_note": (
            "ALL records are SIMULATED. Nothing in this campaign is "
            "MEASURED OAK-D data and no real-world accuracy follows."
        ),
        "config": config.to_dict(),
        "protocol_manifest": protocol.manifest(),
        "completion": protocol.completion(),
        "experiment": experiment_report,
        "coverage": coverage,
        "availability": availability,
        "repeatability": repeatability,
        "outliers": outliers,
        "coverage_acceptance": coverage_gate,
        "acceptance": acceptance.to_dict(),
        "validation_classification": {
            "software_workflow": "SOFTWARE-VALIDATED (unit tests)",
            "synthetic_bias_recovery": "SIMULATION-VALIDATED",
            "real_oak_d_hardware": "NOT VALIDATED (Phase 9 BLOCKED)",
            "real_distance_accuracy": "NOT VALIDATED",
            "calibration_generalization_real_world": "NOT VALIDATED",
            "system_navigation": "NOT VALIDATED",
            "safety": "NOT VALIDATED",
        },
    }
    markdown = render_experiment_markdown(experiment_report)

    return CampaignResult(
        config=config,
        protocol=protocol,
        calibration_records=calibration_records,
        validation_records=validation_records,
        all_raw_records=raw_snapshot,
        experiment=experiment,
        frozen_model=frozen,
        split_disjoint=True,
        coverage=coverage,
        acceptance=acceptance,
        report=campaign_report,
        markdown=markdown,
    )


def campaign_report_json(result: CampaignResult) -> str:
    """Deterministic JSON serialization of the campaign report (§19)."""
    return json.dumps(result.report, sort_keys=True, indent=2)
