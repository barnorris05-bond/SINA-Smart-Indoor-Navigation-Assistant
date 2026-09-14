"""
calibration/ — Phase 10 PREPARATION: ground-truth dataset and
calibration experiment framework.

SOFTWARE-ONLY. All synthetic data in tests is SIMULATED; real OAK-D
measurement calibration remains blocked pending Phase 9 hardware
validation. Nothing here claims distance accuracy, calibration
generalization, or safety.

Workflow this package prepares:

    KNOWN PHYSICAL DISTANCE (operator ground truth)
        -> REAL OAK-D MEASUREMENT (MEASURED provenance; Phase 9+)
        -> MeasurementRecord (evaluation/record.py, reused verbatim)
        -> ground-truth attachment (evaluation/recorder.attach_truth)
        -> CalibrationExperiment.run()
               baseline -> candidate fit on CALIBRATION data only
               -> frozen model -> INDEPENDENT validation metrics
        -> experiment report (deterministic JSON)

Core invariants:

- Calibration and validation observations are DISJOINT
  (metrics.assert_disjoint raises on overlap; never auto-removed).
- Raw predicted distances are preserved next to calibrated values.
- The validation stage cannot refit: FrozenCalibrationModel has no fit.
- Provenance is copied verbatim; a calibration transform NEVER turns
  SIMULATED into MEASURED.
"""

from calibration.session import ExperimentSession
from calibration.plan import TargetDistancePlan, DEFAULT_EXPERIMENT_TARGETS_M
from calibration.coverage import (
    coverage_report,
    availability_breakdown,
)
from calibration.diagnostics import (
    repeatability_analysis,
    outlier_report,
    filter_outliers,
    FilterResult,
    RepeatabilityGroup,
)
from calibration.models import (
    CalibrationError,
    CalibrationModel,
    IdentityCalibration,
    AffineCalibration,
    FrozenCalibrationModel,
    CandidateEvaluation,
    CandidateComparison,
    compare_candidates,
)
from calibration.experiment import (
    CalibratedObservation,
    CalibrationExperiment,
    ExperimentResult,
    apply_calibration,
    evaluate_calibrated,
)
from calibration.reporting import (
    build_experiment_report,
    render_experiment_markdown,
)

__all__ = [
    "ExperimentSession",
    "TargetDistancePlan",
    "DEFAULT_EXPERIMENT_TARGETS_M",
    "coverage_report",
    "availability_breakdown",
    "repeatability_analysis",
    "outlier_report",
    "filter_outliers",
    "FilterResult",
    "RepeatabilityGroup",
    "CalibrationError",
    "CalibrationModel",
    "IdentityCalibration",
    "AffineCalibration",
    "FrozenCalibrationModel",
    "CandidateEvaluation",
    "CandidateComparison",
    "compare_candidates",
    "CalibratedObservation",
    "CalibrationExperiment",
    "ExperimentResult",
    "apply_calibration",
    "evaluate_calibrated",
    "build_experiment_report",
    "render_experiment_markdown",
]
