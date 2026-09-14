"""
calibration/experiment.py

CALIBRATION EXPERIMENT ORCHESTRATOR (Phase 10 preparation).

Runs the complete gated workflow over MeasurementRecord datasets:

    CALIBRATION set                      VALIDATION set
    (operator-identified)                (independent observations)
         |                                     |
         v                                     |
    assert_disjoint()  -- overlap -> FAIL      |
         |                                     |
         v                                     |
    baseline metrics (raw, uncalibrated) §10   |
         |                                     |
         v                                     |
    candidate fit + selection                  |
    (CALIBRATION data ONLY) §19                |
         |                                     |
         v                                     |
    FROZEN model -------------------------> apply
                                               |
                                               v
                                    calibrated validation metrics
                                    (raw baseline reported ALONGSIDE §10)

Honesty rules enforced here:

- Raw predicted distances are preserved: CalibratedObservation carries
  raw AND calibrated values plus their provenances (§8).
- Baseline metrics are always part of the result (§10) — a calibrated
  number can never be presented alone.
- Disjointness is checked BEFORE any fitting; overlap fails loudly and
  no records are auto-removed (§3).
- Provenance: identity keeps MEASURED/SIMULATED verbatim; fitted
  transforms produce explicit *_CALIBRATED provenances that can never
  masquerade as sensor measurements (§7/§22).
- Deterministic: no wall clock, no randomness anywhere.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from evaluation.metrics import (
    EvaluationSummary,
    assert_disjoint,
    check_overlap,
    evaluate,
    validation_class,
)
from evaluation.record import MeasurementRecord

from calibration.models import (
    CalibrationModel,
    FrozenCalibrationModel,
    IdentityCalibration,
    compare_candidates,
    usable_provenance_for_scoring,
)


# --------------------------------------------------------------
# Calibrated observation (raw preserved, §8)
# --------------------------------------------------------------

@dataclass(frozen=True)
class CalibratedObservation:
    """One observation with raw AND calibrated values side by side."""

    scene_id: str
    frame_id: int
    track_id: Optional[int]
    label: str
    ground_truth_m: Optional[float]
    raw_predicted_m: Optional[float]
    raw_provenance: str
    calibrated_predicted_m: Optional[float]
    calibrated_provenance: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "track_id": self.track_id,
            "label": self.label,
            "ground_truth_m": self.ground_truth_m,
            "raw_predicted_m": self.raw_predicted_m,
            "raw_provenance": self.raw_provenance,
            "calibrated_predicted_m": self.calibrated_predicted_m,
            "calibrated_provenance": self.calibrated_provenance,
        }


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------

def apply_calibration(
    records: Sequence[MeasurementRecord],
    model: FrozenCalibrationModel,
) -> List[CalibratedObservation]:
    """
    Map records through the FROZEN model into CalibratedObservations.

    Raw values and provenances are copied verbatim; calibrated
    provenance comes from the model's explicit mapping. Records without
    a predicted value keep calibrated_predicted_m=None (no fabrication).
    """
    out: List[CalibratedObservation] = []
    for r in records:
        calibrated = (
            None if r.predicted_distance_m is None
            else model.transform(r.predicted_distance_m)
        )
        out.append(CalibratedObservation(
            scene_id=r.scene_id,
            frame_id=r.frame_id,
            track_id=r.track_id,
            label=r.label,
            ground_truth_m=r.ground_truth_distance_m,
            raw_predicted_m=r.predicted_distance_m,
            raw_provenance=r.distance_provenance,
            calibrated_predicted_m=calibrated,
            calibrated_provenance=model.calibrated_provenance_for(
                r.distance_provenance
            ),
        ))
    return out


def evaluate_calibrated(
    observations: Sequence[CalibratedObservation],
    provenances: Optional[Sequence[str]] = None,
) -> EvaluationSummary:
    """
    Accuracy summary over CalibratedObservations, reusing
    evaluation.metrics math via lightweight record adapters.

    provenances restricts to SOURCE-vocabulary strings (MEASURED /
    SIMULATED / UNAVAILABLE / STALE) — the adapter maps each
    observation's explicit *_CALIBRATED label to its source vocabulary
    for scoring, so filtering behaves identically before and after
    calibration. None = all observations (invalid statuses counted in
    invalid_rate, never in accuracy).
    """
    records = [_to_record(o) for o in observations]
    allowed = None if provenances is None else set(provenances)
    selected = [
        r for r in records
        if allowed is None or r.distance_provenance in allowed
    ]
    return evaluate(selected)


def _to_record(o: CalibratedObservation) -> MeasurementRecord:
    """Adapter for metrics math: calibrated provenance mapped to the
    usable scoring vocabulary (see models.usable_provenance_for_scoring).
    Data products keep the explicit *_CALIBRATED labels."""
    return MeasurementRecord(
        schema_version=1,
        scene_id=o.scene_id,
        frame_id=o.frame_id,
        timestamp=0.0,
        track_id=o.track_id,
        label=o.label,
        confidence=0.0,
        bbox=(0, 0, 0, 0),
        region="CENTER",
        predicted_distance_m=o.calibrated_predicted_m,
        distance_provenance=usable_provenance_for_scoring(o.calibrated_provenance),
        distance_source=None,
        ground_truth_distance_m=o.ground_truth_m,
    )


# --------------------------------------------------------------
# The experiment
# --------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentResult:
    """
    Complete, deterministic result of one calibration experiment.

    Baseline (raw) and calibrated metrics are ALWAYS both present
    (§10); calibration and validation performance are reported under
    separate keys and never merged.
    """

    n_calibration: int
    n_validation: int
    baseline_validation: EvaluationSummary      # raw model, validation data
    baseline_calibration: EvaluationSummary     # raw model, calibration data
    selected_model: str
    selection_rule: str
    candidate_comparison: Dict[str, object]     # §19 audit trail
    calibration_metrics: EvaluationSummary      # frozen model, calibration data
    validation_metrics: EvaluationSummary       # frozen model, validation data
    validation_class: str
    observations: Optional[List[CalibratedObservation]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_calibration": self.n_calibration,
            "n_validation": self.n_validation,
            "baseline_calibration": self.baseline_calibration.to_dict(),
            "baseline_validation": self.baseline_validation.to_dict(),
            "selected_model": self.selected_model,
            "selection_rule": self.selection_rule,
            "candidate_comparison": self.candidate_comparison,
            "calibration_metrics": self.calibration_metrics.to_dict(),
            "validation_metrics": self.validation_metrics.to_dict(),
            "validation_class": self.validation_class,
        }


class CalibrationExperiment:
    """
    Gated calibration experiment runner.

    Usage:

        experiment = CalibrationExperiment(candidates=[...])
        result = experiment.run(calibration_records, validation_records)

    The default candidate list is [IdentityCalibration] — i.e. with no
    explicit candidates the experiment simply establishes the
    UNCALIBRATED BASELINE (§10) and reports it honestly.
    """

    def __init__(
        self,
        candidates: Optional[Sequence[CalibrationModel]] = None,
        keep_observations: bool = False,
    ) -> None:
        self._candidates: List[CalibrationModel] = (
            list(candidates) if candidates is not None else [IdentityCalibration()]
        )
        self._keep_observations = keep_observations

    def run(
        self,
        calibration_records: Sequence[MeasurementRecord],
        validation_records: Sequence[MeasurementRecord],
    ) -> ExperimentResult:
        # ---- Gate 1: disjointness BEFORE any fitting (§3) ----------
        overlap = check_overlap(calibration_records, validation_records)
        assert_disjoint(calibration_records, validation_records)  # raises

        # ---- Baseline FIRST (§10): raw model on both sets ----------
        baseline_calib = evaluate(calibration_records)
        baseline_val = evaluate(validation_records)

        # ---- Candidate fit + selection on CALIBRATION data (§19) ---
        comparison, frozen = compare_candidates(
            self._candidates, calibration_records
        )

        # ---- Frozen application to both sets ------------------------
        # The provenance filter in evaluate_calibrated maps the explicit
        # *_CALIBRATED labels to the scoring vocabulary, so both fitted
        # and identity models are scored over their full record sets
        # (invalid statuses land in invalid_rate, never in accuracy).
        calib_obs = apply_calibration(calibration_records, frozen)
        val_obs = apply_calibration(validation_records, frozen)
        calibration_metrics = evaluate_calibrated(calib_obs)
        validation_metrics = evaluate_calibrated(val_obs)

        # ---- Honest overall classification --------------------------
        val_provenances = sorted({r.distance_provenance for r in validation_records})
        vclass = validation_class(val_provenances)

        return ExperimentResult(
            n_calibration=len(calibration_records),
            n_validation=len(validation_records),
            baseline_calibration=baseline_calib,
            baseline_validation=baseline_val,
            selected_model=frozen.name,
            selection_rule=comparison.selection_rule,
            candidate_comparison=comparison.to_dict(),
            calibration_metrics=calibration_metrics,
            validation_metrics=validation_metrics,
            validation_class=vclass,
            observations=(val_obs if self._keep_observations else None),
        )

    @staticmethod
    def overlap_report(
        calibration_records: Sequence[MeasurementRecord],
        validation_records: Sequence[MeasurementRecord],
    ) -> Dict[str, object]:
        """Descriptive overlap summary (for pre-run inspection)."""
        report = check_overlap(calibration_records, validation_records)
        return {
            "exact_overlaps": len(report.exact_overlaps),
            "shared_scene_ids": list(report.shared_scene_ids),
            "is_clean": report.is_clean,
        }
