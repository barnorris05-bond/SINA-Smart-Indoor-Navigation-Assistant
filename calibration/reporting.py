"""
calibration/reporting.py

CALIBRATION EXPERIMENT REPORT (Phase 10 preparation).

Assembles a deterministic JSON report from an ExperimentResult plus
coverage/diagnostic analyses. It computes NO new statistics: it only
structures, orders, and renders what analysis produced — same inputs,
byte-identical output.

Report sections (§21):

    metadata, sample_counts, provenance, distance_coverage,
    scene_coverage, label_coverage, invalid_rate, raw_baseline,
    calibration_model, calibration_performance, validation_performance,
    error_distribution, repeatability, outliers, limitations,
    validation_classification

CALIBRATION PERFORMANCE (fit quality) and VALIDATION PERFORMANCE
(generalization) are separate, clearly-labeled sections — calibration
numbers are never presented as validation numbers.
"""

from typing import Dict, Optional, Sequence

import numpy as np

from evaluation.record import MeasurementRecord

from calibration.diagnostics import (
    FilterResult,
    outlier_report,
    repeatability_analysis,
)
from calibration.experiment import ExperimentResult
from calibration.plan import TargetDistancePlan

LIMITATIONS = (
    "Phase 10 preparation is SOFTWARE-ONLY: all framework validation to "
    "date uses SIMULATED fixtures. Real OAK-D measurement calibration "
    "requires Phase 9 hardware validation (currently BLOCKED), real "
    "MEASURED observations, and independent operator ground truth. "
    "Calibration performance shown here (if any) describes fit quality "
    "on calibration data ONLY and does not demonstrate generalization. "
    "No navigation effectiveness, distance accuracy, or safety claims "
    "are supported by this framework by itself."
)


def build_experiment_report(
    result: ExperimentResult,
    calibration_records: Optional[Sequence[MeasurementRecord]] = None,
    validation_records: Optional[Sequence[MeasurementRecord]] = None,
    requested_targets_m: Optional[Sequence[float]] = None,
    outlier_filter: Optional[FilterResult] = None,
) -> Dict[str, object]:
    """
    Structured, deterministic experiment report (JSON-ready dict).

    Records are optional enrichment inputs for coverage/diagnostic
    sections; the report contains no wall-clock fields, so identical
    inputs yield identical output.
    """
    report: Dict[str, object] = {
        "metadata": {
            "phase": "10-preparation (software-only)",
            "hardware_validation": "BLOCKED - Phase 9 Checkpoint A not performed",
            "calibrated": False,
            "selected_model": result.selected_model,
            "selection_rule": result.selection_rule,
        },
        "sample_counts": {
            "calibration": result.n_calibration,
            "validation": result.n_validation,
        },
        "provenance": {
            "calibration": result.baseline_calibration.provenances,
            "validation": result.baseline_validation.provenances,
            "note": (
                "Provenance is recorded verbatim from the capture pipeline; "
                "calibrated values carry explicit *_CALIBRATED labels and "
                "never masquerade as raw MEASURED sensor data."
            ),
        },
        "raw_baseline": {
            "calibration": result.baseline_calibration.to_dict(),
            "validation": result.baseline_validation.to_dict(),
            "note": "UNCALIBRATED BASELINE - always reported (§10).",
        },
        "calibration_model": result.candidate_comparison,
        "calibration_performance": {
            "frozen_model_on_calibration_data": result.calibration_metrics.to_dict(),
            "note": (
                "Fit quality on the CALIBRATION set - not a generalization "
                "claim."
            ),
        },
        "validation_performance": {
            "frozen_model_on_validation_data": result.validation_metrics.to_dict(),
            "validation_class": result.validation_class,
            "note": (
                "Frozen model applied to the INDEPENDENT validation set; "
                "no refit is possible by construction."
            ),
        },
        "limitations": LIMITATIONS,
        "validation_classification": {
            "software": "SOFTWARE-VALIDATED (unit tests)",
            "synthetic_workflow": "SIMULATION-VALIDATED (simulated fixtures)",
            "real_oak_d_hardware": "NOT VALIDATED (Phase 9 BLOCKED)",
            "real_distance_accuracy": "NOT VALIDATED",
            "calibration_generalization": "NOT VALIDATED",
            "system_navigation": "NOT VALIDATED",
            "safety": "NOT VALIDATED",
        },
    }

    # ---- Coverage sections (need records) --------------------------
    if calibration_records is not None and validation_records is not None:
        all_records = list(calibration_records) + list(validation_records)
        report["distance_coverage"] = _coverage_summary(
            calibration_records, validation_records, requested_targets_m
        )
        report["scene_coverage"] = {
            "calibration_scenes": sorted({r.scene_id for r in calibration_records}),
            "validation_scenes": sorted({r.scene_id for r in validation_records}),
            "shared_scenes": sorted(
                {r.scene_id for r in calibration_records}
                & {r.scene_id for r in validation_records}
            ),
            "note": (
                "Scene diversity across the split strengthens the future "
                "generalization argument; it is descriptive only."
            ),
        }
        report["label_coverage"] = {
            "calibration_labels": sorted({r.label for r in calibration_records}),
            "validation_labels": sorted({r.label for r in validation_records}),
        }
        report["invalid_rate"] = {
            "calibration": result.baseline_calibration.invalid_rate,
            "validation": result.baseline_validation.invalid_rate,
            "by_distance": _availability(all_records),
        }
        report["error_distribution"] = outlier_report(all_records)
        report["repeatability"] = [
            g.to_dict() for g in repeatability_analysis(all_records, min_samples=3)
        ]
        report["outliers"] = outlier_report(all_records)["largest_absolute_errors"]
        report["outlier_filtering"] = (
            outlier_filter.to_dict() if outlier_filter is not None else None
        )
    return report


def _coverage_summary(
    calibration_records: Sequence[MeasurementRecord],
    validation_records: Sequence[MeasurementRecord],
    requested_targets_m: Optional[Sequence[float]],
) -> Dict[str, object]:
    def _stats(rows: Sequence[MeasurementRecord]) -> Dict[str, object]:
        truths = [
            r.ground_truth_distance_m for r in rows
            if r.ground_truth_distance_m is not None
            and r.ground_truth_distance_m > 0
        ]
        if not truths:
            return {"n": 0}
        arr = np.array(truths, dtype=float)
        return {
            "n": len(truths),
            "min_m": float(np.min(arr)),
            "median_m": float(np.median(arr)),
            "max_m": float(np.max(arr)),
            "unique": int(len(set(truths))),
        }

    summary: Dict[str, object] = {
        "calibration": _stats(calibration_records),
        "validation": _stats(validation_records),
        "note": (
            "Requested targets (plan) and observed ground truth are "
            "different quantities; both are reported separately."
        ),
    }
    if requested_targets_m is not None:
        summary["requested_targets_m"] = sorted(set(requested_targets_m))
    return summary


def _availability(records: Sequence[MeasurementRecord]) -> Dict[str, object]:
    from calibration.coverage import availability_breakdown
    return availability_breakdown(records)["by_distance"]


def render_experiment_markdown(report: Dict[str, object]) -> str:
    """
    Human-readable Markdown rendering of the experiment report.
    Consumes the structured dict only (no recomputation).
    """
    lines: list[str] = []
    meta = report.get("metadata", {})
    lines.append("# Calibration experiment report (Phase 10 preparation)")
    lines.append("")
    lines.append(f"Phase: **{meta.get('phase', 'n/a')}**")
    lines.append(f"Hardware validation: **{meta.get('hardware_validation', 'n/a')}**")
    lines.append("")

    counts = report.get("sample_counts", {})
    lines.append("## Samples")
    lines.append(f"- calibration: {counts.get('calibration', 'n/a')}")
    lines.append(f"- validation: {counts.get('validation', 'n/a')}")
    lines.append("")

    base = report.get("raw_baseline", {})
    for role in ("calibration", "validation"):
        s = base.get(role, {})
        lines.append(
            f"- raw baseline {role}: "
            f"MAE={_f(s.get('mae_m'))} m, p95={_f(s.get('p95_abs_error_m'))} m, "
            f"invalid_rate={_f(s.get('invalid_rate'))}"
        )
    if base.get("note"):
        lines.append("")
        lines.append(str(base["note"]))
    lines.append("")

    model = report.get("calibration_model", {})
    lines.append("## Calibration model")
    lines.append(f"- selected: **{model.get('selected', 'n/a')}**")
    lines.append(f"- rule: {model.get('selection_rule', 'n/a')}")
    lines.append("")
    for ev in model.get("evaluations", []):
        lines.append(
            f"  - {ev.get('name')}: MAE={_f(ev.get('mae_m'))} m "
            f"(n={ev.get('n_calibration_samples')})"
        )
    lines.append("")

    cal = report.get("calibration_performance", {})
    val = report.get("validation_performance", {})
    cs = cal.get("frozen_model_on_calibration_data", {})
    vs = val.get("frozen_model_on_validation_data", {})
    lines.append("## Performance (frozen model)")
    lines.append(
        f"- CALIBRATION (fit quality): MAE={_f(cs.get('mae_m'))} m, "
        f"p95={_f(cs.get('p95_abs_error_m'))} m"
    )
    lines.append(
        f"- VALIDATION (independent): MAE={_f(vs.get('mae_m'))} m, "
        f"p95={_f(vs.get('p95_abs_error_m'))} m, "
        f"class={val.get('validation_class', 'n/a')}"
    )
    lines.append("")

    lines.append("## Limitations")
    lines.append(str(report.get("limitations", "")))
    lines.append("")
    lines.append("## Validation classification")
    for key, value in report.get("validation_classification", {}).items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def _f(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"
