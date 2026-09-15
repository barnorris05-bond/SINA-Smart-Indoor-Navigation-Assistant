"""
evaluation/metrics.py

DISTANCE ACCURACY METRICS (pure software, no I/O, deterministic).

All metrics compare predicted distance against ground truth over
MeasurementRecord lists. Definitions (fixed, documented):

    absolute_error  = abs(predicted - ground_truth)
    relative_error  = absolute_error / ground_truth        (gt > 0 enforced)
    MAE             = mean(absolute_error)
    RMSE            = sqrt(mean(absolute_error ** 2))
    median_abs_err  = median(absolute_error)
    max_abs_err     = max(absolute_error)
    median_rel_err  = median(relative_error)
    p95_abs_err     = 95th percentile of absolute_error (linear interp.)

INVALID MEASUREMENT POLICY (milestone requirement - no silent zeros):

A record is VALID for accuracy metrics iff ALL of:
    - predicted_distance_m is not None
    - distance_provenance in ("MEASURED", "SIMULATED")   (usable vocab)
    - predicted_distance_m > 0
    - ground_truth_distance_m is not None
    - ground_truth_distance_m > 0
Everything else is INVALID: excluded from accuracy metrics and counted
in invalid_rate = n_invalid / n_total. UNAVAILABLE / STALE records and
missing ground truth therefore never become zero errors.

PROVENANCE POLICY:

evaluate() never silently combines provenances: the summary lists
exactly which provenances contributed, and validation_class() states
what the data can honestly claim:
    {"SIMULATED"}          -> SIMULATION-VALIDATED
    {"MEASURED"}           -> MEASUREMENT-VALIDATED (per recorded provenance)
    mixed                  -> MIXED - split by provenance before comparing
    no usable rows         -> NO USABLE MEASUREMENTS
("MEASUREMENT-VALIDATED" refers to provenance recorded from a real OAK-D
capture session; it is only as honest as the capture pipeline that
stamped it.)

CALIBRATION / VALIDATION SEPARATION:

check_overlap() detects whether a validation set shares observations
(scene_id, frame_id, track_id) with a calibration set. Exact overlap is
an error condition (assert_disjoint raises) - tuning and validation on
the same measurements is how accuracy gets silently overstated.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from evaluation.record import MeasurementRecord, VALID_PROVENANCES

USABLE_PROVENANCES = ("MEASURED", "SIMULATED")


def is_valid_record(rec: MeasurementRecord) -> bool:
    """See module docstring: the exact validity rule."""
    if rec.predicted_distance_m is None or rec.ground_truth_distance_m is None:
        return False
    if rec.distance_provenance not in USABLE_PROVENANCES:
        return False
    if rec.predicted_distance_m <= 0 or rec.ground_truth_distance_m <= 0:
        return False
    return True


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate accuracy of one evaluation run (immutable)."""

    n_total: int
    n_valid: int
    n_invalid: int
    invalid_rate: float
    provenances: Tuple[str, ...]          # provenances present in the data
    validation_class: str                 # see validation_class()
    mae_m: Optional[float]
    rmse_m: Optional[float]
    median_abs_error_m: Optional[float]
    max_abs_error_m: Optional[float]
    median_relative_error: Optional[float]
    p95_abs_error_m: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_total": self.n_total,
            "n_valid": self.n_valid,
            "n_invalid": self.n_invalid,
            "invalid_rate": self.invalid_rate,
            "provenances": list(self.provenances),
            "validation_class": self.validation_class,
            "mae_m": self.mae_m,
            "rmse_m": self.rmse_m,
            "median_abs_error_m": self.median_abs_error_m,
            "max_abs_error_m": self.max_abs_error_m,
            "median_relative_error": self.median_relative_error,
            "p95_abs_error_m": self.p95_abs_error_m,
        }


def validation_class(provenances: Iterable[str]) -> str:
    """What a set of contributing provenances can honestly claim."""
    usable = {p for p in provenances if p in USABLE_PROVENANCES}
    if not usable:
        return "NO USABLE MEASUREMENTS"
    if usable == {"SIMULATED"}:
        return "SIMULATION-VALIDATED"
    if usable == {"MEASURED"}:
        return "MEASUREMENT-VALIDATED (per recorded provenance)"
    return "MIXED PROVENANCES - split by provenance before comparing"


def evaluate(
    records: Sequence[MeasurementRecord],
    provenances: Optional[Sequence[str]] = None,
) -> EvaluationSummary:
    """
    Compute the accuracy summary for a record sequence.

    provenances optionally restricts the evaluation to specific
    provenance strings (e.g. ("MEASURED",)); when None, ALL records are
    used - the summary always lists which provenances contributed, so
    mixing MEASURED and SIMULATED is visible, never silent.
    """
    selected: List[MeasurementRecord] = list(records)
    if provenances is not None:
        allowed = set(provenances)
        selected = [r for r in selected if r.distance_provenance in allowed]

    present = sorted({r.distance_provenance for r in selected})
    valid = [r for r in selected if is_valid_record(r)]
    n_total = len(selected)
    n_invalid = n_total - len(valid)

    if not valid:
        return EvaluationSummary(
            n_total=n_total,
            n_valid=0,
            n_invalid=n_invalid,
            invalid_rate=(n_invalid / n_total) if n_total else 0.0,
            provenances=tuple(present),
            validation_class=validation_class(present),
            mae_m=None,
            rmse_m=None,
            median_abs_error_m=None,
            max_abs_error_m=None,
            median_relative_error=None,
            p95_abs_error_m=None,
        )

    pred = np.array([r.predicted_distance_m for r in valid], dtype=float)
    gt = np.array([r.ground_truth_distance_m for r in valid], dtype=float)
    abs_err = np.abs(pred - gt)                      # gt > 0 guaranteed
    rel_err = abs_err / gt

    return EvaluationSummary(
        n_total=n_total,
        n_valid=len(valid),
        n_invalid=n_invalid,
        invalid_rate=n_invalid / n_total if n_total else 0.0,
        provenances=tuple(present),
        validation_class=validation_class(present),
        mae_m=float(np.mean(abs_err)),
        rmse_m=float(np.sqrt(np.mean(abs_err ** 2))),
        median_abs_error_m=float(np.median(abs_err)),
        max_abs_error_m=float(np.max(abs_err)),
        median_relative_error=float(np.median(rel_err)),
        p95_abs_error_m=float(np.percentile(abs_err, 95.0)),
    )


# ----------------------------------------------------------
# Calibration / validation separation
# ----------------------------------------------------------

@dataclass(frozen=True)
class OverlapReport:
    """Result of comparing a calibration set with a validation set."""

    exact_overlaps: Tuple[Tuple[str, int, Optional[int]], ...]
    shared_scene_ids: Tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not self.exact_overlaps


def check_overlap(
    calibration: Sequence[MeasurementRecord],
    validation: Sequence[MeasurementRecord],
) -> OverlapReport:
    """
    Compare calibration and validation records by observation identity
    (scene_id, frame_id, track_id). Identical triples mean the SAME
    observation would be used for both tuning and validation.
    """
    calib_keys = {r.identity_key() for r in calibration}
    val_keys = {r.identity_key() for r in validation}
    overlaps = tuple(sorted(calib_keys & val_keys))
    shared_scenes = tuple(sorted(
        {r.scene_id for r in calibration}
        & {r.scene_id for r in validation}
    ))
    return OverlapReport(exact_overlaps=overlaps, shared_scene_ids=shared_scenes)


def assert_disjoint(
    calibration: Sequence[MeasurementRecord],
    validation: Sequence[MeasurementRecord],
) -> OverlapReport:
    """Raise ValueError when calibration and validation share observations."""
    report = check_overlap(calibration, validation)
    if report.exact_overlaps:
        raise ValueError(
            "calibration/validation overlap detected "
            f"({len(report.exact_overlaps)} shared observations, e.g. "
            f"{report.exact_overlaps[0]!r}) - never tune and validate on "
            "the same measurements"
        )
    return report
