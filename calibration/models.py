"""
calibration/models.py

CALIBRATION MODEL INTERFACE (Phase 10 preparation).

Minimal, deterministic, math-free-by-default framework for candidate
corrections of predicted distances (§9-§12, §19-§20):

    fit(calibration_records) -> model        # CALIBRATION data only
    model.transform(predicted_m) -> float    # frozen afterwards
    evaluate(validation_records)             # separate observations

Rules enforced structurally:

- FIT/VISIBILITY ISOLATION: calibration models receive ONLY
  calibration records. Nothing in the fit path ever sees validation
  data; the experiment object passes disjoint record lists (enforced
  with metrics.assert_disjoint BEFORE fitting).
- FIT/TRANSFORM SEPARATION: the object returned by fit() —
  FrozenCalibrationModel — exposes transform() and CANNOT refit: the
  validation stage is read-only with respect to calibration parameters
  (§20). Fit-time diagnostics (CandidateEvaluation) live on the
  pre-freeze object only.
- RAW PRESERVATION (§8): transform() maps one raw predicted distance
  to a calibrated value; it never rewrites records. The experiment
  keeps raw and calibrated values side by side in
  CalibratedObservation.
- PROVENANCE (§22): an identity transform preserves provenance
  verbatim; a fitted transform marks calibrated provenance
  "MEASURED_CALIBRATED" — explicitly NOT the raw MEASURED vocabulary,
  so a corrected value can never masquerade as a sensor measurement.
- CANDIDATE SELECTION (§19): compare_candidates() evaluates every
  candidate on the calibration set ONLY, with a deterministic
  documented tie-break rule (lowest MAE; tie -> lowest parameter
  count; tie -> earlier registration). Selection never uses
  validation data.

Built-in candidates (§9: start simple, no ML):
  IdentityCalibration — no-op; defines the UNCALIBRATED BASELINE (§10)
  AffineCalibration   — least-squares y = a*x + b (closed form)
Nonlinear models can be added later behind the same interface; no
model is assumed appropriate before real data exists.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluation.metrics import evaluate, is_valid_record
from evaluation.record import MeasurementRecord

USABLE_PROVENANCES = ("MEASURED", "SIMULATED")

# Provenance for CALIBRATED values: deliberately OUTSIDE the original
# four-value vocabulary so a transformed value can never be confused
# with a raw sensor measurement (or with a simulated one).
CALIBRATED_PROVENANCE = "MEASURED_CALIBRATED"
CALIBRATED_PROVENANCE_SIMULATED = "SIMULATED_CALIBRATED"


class CalibrationError(ValueError):
    """Raised for invalid calibration data or degenerate fits."""


# --------------------------------------------------------------
# Model interface
# --------------------------------------------------------------

class CalibrationModel:
    """
    A FITTABLE calibration candidate.

    Subclasses implement _fit(rows) and transform(). fit() normalizes
    input handling (rejection of unusable rows) once, here.
    """

    name: str = "CalibrationModel"
    #: parameters determined by fit; models must set this after fitting
    n_parameters: int = 0

    # -- fitting ---------------------------------------------------

    def fit(self, calibration_records: Sequence[MeasurementRecord]) -> "FrozenCalibrationModel":
        """
        Fit on CALIBRATION records and return the FROZEN model.

        Only valid records (usable provenance, positive predicted and
        ground-truth values) contribute; if fewer than min_samples
        remain, CalibrationError is raised — the framework surfaces
        narrow coverage loudly instead of fitting noise (§12).
        """
        raise NotImplementedError

    # -- application ------------------------------------------------

    def transform(self, predicted_distance_m: float) -> float:
        """Map one RAW predicted distance to the calibrated value."""
        raise NotImplementedError

    # -- provenance bookkeeping --------------------------------------

    def calibrated_provenance_for(self, source_provenance: str) -> str:
        """
        Provenance of a transformed value derived from source_provenance.

        Identity keeps the vocabulary verbatim; fitted transforms map
        usable provenances to explicit *_CALIBRATED labels (never back
        to MEASURED/SIMULATED), and non-usable provenances through
        unchanged — availability status is not transformed by a
        distance correction.
        """
        if source_provenance == "UNAVAILABLE" or source_provenance == "STALE":
            return source_provenance
        if source_provenance == "MEASURED":
            return (
                "MEASURED" if self.is_identity else CALIBRATED_PROVENANCE
            )
        if source_provenance == "SIMULATED":
            return (
                "SIMULATED" if self.is_identity
                else CALIBRATED_PROVENANCE_SIMULATED
            )
        return source_provenance

    @property
    def is_identity(self) -> bool:
        return False

    # -- shared fit helpers -------------------------------------------

    @staticmethod
    def _fit_rows(
        records: Sequence[MeasurementRecord], min_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract (predicted, ground_truth) arrays from valid records.

        Non-finite values (nan/inf) are rejected IN ADDITION to the
        is_valid_record gate (Phase 12 §8): inf > 0 and nan comparisons
        would otherwise slip into the least-squares fit and poison the
        parameters. Invalid values are excluded from the fit, never
        clamped; coverage stays visible via the sample count.
        """
        valid = [
            r for r in records
            if is_valid_record(r)
            and math.isfinite(r.predicted_distance_m)
            and math.isfinite(r.ground_truth_distance_m)
        ]
        if len(valid) < min_samples:
            raise CalibrationError(
                f"insufficient calibration samples: {len(valid)} valid of "
                f"{len(records)} records, need >= {min_samples} (§12: narrow "
                "coverage must be visible, not fitted through)"
            )
        pred = np.array([r.predicted_distance_m for r in valid], dtype=float)
        gt = np.array([r.ground_truth_distance_m for r in valid], dtype=float)
        return pred, gt

    def _check_fitted(self) -> None:
        if getattr(self, "_params", None) is None:
            raise CalibrationError(
                f"{type(self).__name__} has no fitted parameters — "
                "call fit() first"
            )


# --------------------------------------------------------------
# Candidate 0: identity (the uncalibrated baseline)
# --------------------------------------------------------------

class IdentityCalibration(CalibrationModel):
    """No-op correction. Evaluating it IS the baseline (§10)."""

    name = "identity"

    def __init__(self) -> None:
        self._params: Tuple[float, ...] = ()

    @property
    def is_identity(self) -> bool:
        return True

    def fit(
        self, calibration_records: Sequence[MeasurementRecord]
    ) -> "FrozenCalibrationModel":
        # Identity has nothing to estimate; still enforce the same
        # visibility of narrow coverage as fitted models (§12).
        self._fit_rows(calibration_records, min_samples=1)
        self._params = ()
        return FrozenCalibrationModel(self)

    def transform(self, predicted_distance_m: float) -> float:
        return float(predicted_distance_m)


# --------------------------------------------------------------
# Candidate 1: affine least squares
# --------------------------------------------------------------

class AffineCalibration(CalibrationModel):
    """
    Closed-form least-squares y = a*x + b over calibration pairs
    (predicted -> ground truth). The simplest mathematically justified
    correction beyond identity; no ML (§9, §11).
    """

    name = "affine"
    n_parameters = 2

    def __init__(self) -> None:
        self._params: Optional[Tuple[float, float]] = None

    def fit(
        self, calibration_records: Sequence[MeasurementRecord]
    ) -> "FrozenCalibrationModel":
        pred, gt = self._fit_rows(calibration_records, min_samples=2)
        # Closed-form least squares via polyfit (deterministic).
        a, b = (float(v) for v in np.polyfit(pred, gt, 1))
        self._params = (a, b)
        return FrozenCalibrationModel(self)

    def transform(self, predicted_distance_m: float) -> float:
        self._check_fitted()
        a, b = self._params  # type: ignore[misc]
        return a * float(predicted_distance_m) + b


# --------------------------------------------------------------
# Frozen model (validation stage - read-only)
# --------------------------------------------------------------

@dataclass(frozen=True)
class FrozenCalibrationModel:
    """
    A fitted, IMMUTABLE calibration model.

    Exposes transform() and provenance mapping only — no fit(), no
    parameters, no access to calibration data. Applying it to the
    validation set cannot retune anything (§20).
    """

    _model: CalibrationModel

    @property
    def name(self) -> str:
        return self._model.name

    @property
    def n_parameters(self) -> int:
        return self._model.n_parameters

    @property
    def is_identity(self) -> bool:
        return self._model.is_identity

    def transform(self, predicted_distance_m: float) -> float:
        self._model._check_fitted()
        return self._model.transform(predicted_distance_m)

    def calibrated_provenance_for(self, source_provenance: str) -> str:
        return self._model.calibrated_provenance_for(source_provenance)


# --------------------------------------------------------------
# Candidate comparison (§19 - calibration data only)
# --------------------------------------------------------------

@dataclass(frozen=True)
class CandidateEvaluation:
    """One candidate's performance on the CALIBRATION set."""

    name: str
    n_parameters: int
    mae_m: Optional[float]
    rmse_m: Optional[float]
    p95_abs_error_m: Optional[float]
    n_calibration_samples: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "n_parameters": self.n_parameters,
            "mae_m": self.mae_m,
            "rmse_m": self.rmse_m,
            "p95_abs_error_m": self.p95_abs_error_m,
            "n_calibration_samples": self.n_calibration_samples,
        }


@dataclass(frozen=True)
class CandidateComparison:
    """Deterministic candidate ranking over the calibration set."""

    evaluations: Tuple[CandidateEvaluation, ...]
    selected: str
    selection_rule: str = (
        "lowest MAE on CALIBRATION data; tie -> fewer parameters; "
        "tie -> first registered. Validation data is never used for "
        "selection (§19)."
    )

    def to_dict(self) -> Dict[str, object]:
        return {
            "evaluations": [e.to_dict() for e in self.evaluations],
            "selected": self.selected,
            "selection_rule": self.selection_rule,
        }


def compare_candidates(
    candidates: Sequence[CalibrationModel],
    calibration_records: Sequence[MeasurementRecord],
) -> Tuple[CandidateComparison, FrozenCalibrationModel]:
    """
    Fit and score each candidate on the CALIBRATION set, select by the
    documented deterministic rule, and return the comparison plus the
    FROZEN selected model.

    Selection uses calibration metrics only (§19); validation data is
    structurally absent from this function's inputs.
    """
    if not candidates:
        raise CalibrationError("at least one candidate model is required")

    evaluations: List[CandidateEvaluation] = []
    frozen: List[FrozenCalibrationModel] = []
    for cand in candidates:
        fm = cand.fit(calibration_records)
        frozen.append(fm)
        # Score the candidate's TRANSFORMED predictions on the same
        # calibration records (this is fitting performance, §19).
        transformed = [
            _with_calibrated_prediction(
                r, fm.transform, fm.calibrated_provenance_for
            )
            for r in calibration_records
        ]
        summary = evaluate(transformed)
        evaluations.append(CandidateEvaluation(
            name=cand.name,
            n_parameters=fm.n_parameters,
            mae_m=summary.mae_m,
            rmse_m=summary.rmse_m,
            p95_abs_error_m=summary.p95_abs_error_m,
            n_calibration_samples=summary.n_valid,
        ))

    # Deterministic selection: lowest MAE; tie -> fewer parameters;
    # tie -> first registered.
    best = min(
        range(len(evaluations)),
        key=lambda i: (
            evaluations[i].mae_m if evaluations[i].mae_m is not None
            else float("inf"),
            evaluations[i].n_parameters,
            i,
        ),
    )
    comparison = CandidateComparison(
        evaluations=tuple(evaluations), selected=candidates[best].name
    )
    return comparison, frozen[best]


def usable_provenance_for_scoring(provenance: str) -> str:
    """
    Map the *_CALIBRATED labels back to their source vocabulary FOR
    ACCURACY COMPUTATION ONLY: evaluation.metrics' usability gate knows
    the four-value source vocabulary, not the calibrated labels.

    This mapping is scoped to metric math. Data products
    (CalibratedObservation, reports) always carry the explicit
    *_CALIBRATED labels — a corrected value never appears as a raw
    MEASURED sensor measurement anywhere.
    """
    if provenance == CALIBRATED_PROVENANCE:
        return "MEASURED"
    if provenance == CALIBRATED_PROVENANCE_SIMULATED:
        return "SIMULATED"
    return provenance


def _with_calibrated_prediction(
    rec: MeasurementRecord,
    transform,
    provenance_for,
) -> MeasurementRecord:
    """
    Copy a record with its predicted value replaced by the calibrated
    one (and provenance mapped back to the usable vocabulary for
    candidate SCORING — see _usable_provenance_for_scoring). Never
    mutates the input.
    """
    if rec.predicted_distance_m is None:
        return rec
    return MeasurementRecord(
        schema_version=rec.schema_version,
        scene_id=rec.scene_id,
        frame_id=rec.frame_id,
        timestamp=rec.timestamp,
        track_id=rec.track_id,
        label=rec.label,
        confidence=rec.confidence,
        bbox=rec.bbox,
        region=rec.region,
        predicted_distance_m=transform(rec.predicted_distance_m),
        distance_provenance=usable_provenance_for_scoring(
            provenance_for(rec.distance_provenance)
        ),
        distance_source=rec.distance_source,
        ground_truth_distance_m=rec.ground_truth_distance_m,
    )
