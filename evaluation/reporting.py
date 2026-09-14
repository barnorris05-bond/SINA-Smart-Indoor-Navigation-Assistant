"""
evaluation/reporting.py

REPORT GENERATION for the offline analysis layer.

analyze_dataset() runs the structured analysis (analysis.py +
navigation_analysis.py); build_report() assembles a deterministic,
machine-readable JSON report from those results:

    metadata / dataset_summary / validation_triage /
    provenance_summary / overall_metrics / per_label_metrics /
    per_scene_metrics / distance_bins / error_distribution /
    navigation_summary / risk_summary / temporal_summary

- DETERMINISTIC: reports contain no wall-clock fields; the same input
  always produces byte-identical output.
- ROLES: a dataset is CALIBRATION, VALIDATION or UNSPECIFIED - never
  silently treated as validation data.
- evaluate_pair() verifies calibration/validation disjointness with
  metrics.assert_disjoint() BEFORE evaluating; overlapping records fail
  loudly (ValueError) and are never silently removed.
- render_markdown() consumes the structured report - no analysis logic
  is duplicated in rendering.

This tooling does not validate safety and does not validate real-world
distance accuracy by itself.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from evaluation.record import MeasurementRecord
from evaluation.metrics import assert_disjoint, validation_class
from evaluation.analysis import (
    DatasetError,
    dataset_summary,
    distance_bin_analysis,
    error_distribution,
    label_analysis,
    metrics_for,
    provenance_metrics,
    scene_analysis,
    validate_records,
)
from evaluation.navigation_analysis import (
    navigation_stats,
    risk_stats,
    temporal_stats,
)

REPORT_SCHEMA_VERSION = 1

ROLES = ("CALIBRATION", "VALIDATION", "UNSPECIFIED")

LIMITATIONS = (
    "This tooling does not validate safety.",
    "This tooling does not validate real-world distance accuracy by itself.",
    "Metrics over SIMULATED data are SIMULATION-VALIDATED only.",
)


# ------------------------------------------------------------------
# Analysis orchestration
# ------------------------------------------------------------------

def load_jsonl_events(path: Union[str, Path]) -> List[dict]:
    """
    Load an EventLogger JSONL trace (one JSON object per line).
    Malformed lines raise DatasetError with the line number - no silent
    partial reads.
    """
    events: List[dict] = []
    p = Path(path)
    if not p.is_file():
        raise DatasetError(f"events file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("event must be a JSON object")
                events.append(obj)
            except Exception as exc:
                raise DatasetError(
                    f"{p}: malformed event at line {lineno}: {exc}") from exc
    return events


def analyze_dataset(
    records: Sequence[MeasurementRecord],
    events: Optional[Sequence[dict]] = None,
) -> dict:
    """Run the full structured analysis for one dataset."""
    validated = validate_records(records)
    invalid_reasons: Dict[str, int] = {}
    for entry in validated.invalid:
        invalid_reasons[entry.reason] = invalid_reasons.get(entry.reason, 0) + 1

    summary = dataset_summary(records)
    groups = provenance_metrics(records)
    provenance_rows: Dict[str, dict] = {}
    for prov, count in sorted(summary.provenance_counts.items()):
        group = groups.get(prov)
        provenance_rows[prov] = {
            "count": count,
            "n_valid": group.n_valid if group else 0,
            "validation_class": validation_class([prov]),
            "metrics": group.to_dict() if group else None,
        }

    return {
        "dataset_summary": summary.to_dict(),
        "validation_triage": {
            "n_total": len(records),
            "n_invalid": len(validated.invalid),
            "invalid_reasons": dict(sorted(invalid_reasons.items())),
        },
        "provenance_summary": provenance_rows,
        "overall_metrics": metrics_for(records).to_dict(),
        "per_label_metrics": label_analysis(records),
        "per_scene_metrics": scene_analysis(records),
        "distance_bins": distance_bin_analysis(records),
        "error_distribution": error_distribution(records),
        "navigation_summary": (
            navigation_stats(events) if events is not None else None),
        "risk_summary": risk_stats(events) if events is not None else None,
        "temporal_summary": (
            temporal_stats(events) if events is not None else None),
    }


# ------------------------------------------------------------------
# Report assembly
# ------------------------------------------------------------------

def build_report(
    analysis: dict,
    dataset_role: str = "UNSPECIFIED",
    name: str = "sina_evaluation_report",
    source: Optional[str] = None,
) -> dict:
    """
    Assemble the final report from structured analysis results.

    dataset_role must be CALIBRATION, VALIDATION or UNSPECIFIED - the
    report never assumes a dataset is validation data.
    """
    if dataset_role not in ROLES:
        raise ValueError(
            f"dataset_role must be one of {ROLES}, got {dataset_role!r}")
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "name": name,
        "source": source,
        "dataset_role": dataset_role,
        "deterministic": True,
        "limitations": list(LIMITATIONS),
        **analysis,
    }


def evaluate_pair(
    calibration_records: Sequence[MeasurementRecord],
    validation_records: Sequence[MeasurementRecord],
    name: str = "sina_evaluation_report",
) -> dict:
    """
    Evaluate a CALIBRATION and a VALIDATION dataset together.

    Disjointness is asserted FIRST (metrics.assert_disjoint): shared
    observations raise ValueError - overlapping records are never
    silently removed. The returned reports are tagged with their roles.
    """
    assert_disjoint(calibration_records, validation_records)
    return {
        "calibration": build_report(
            analyze_dataset(calibration_records),
            dataset_role="CALIBRATION", name=name),
        "validation": build_report(
            analyze_dataset(validation_records),
            dataset_role="VALIDATION", name=name),
        "disjointness": "VERIFIED (assert_disjoint passed)",
    }


# ------------------------------------------------------------------
# Markdown rendering (consumes the structured report - no new logic)
# ------------------------------------------------------------------

def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(_fmt(c) for c in row) + " |")
    return "\n".join(out)


def render_markdown(report: dict) -> str:
    """Render the structured report as human-readable Markdown."""
    r = report
    lines: List[str] = [f"# {r['name']}", ""]
    lines.append(f"- dataset_role: **{r['dataset_role']}**")
    if r.get("source"):
        lines.append(f"- source: `{r['source']}`")
    lines.append("- deterministic report (no wall-clock fields)")
    lines.append("")
    lines.append("**Limitations:**")
    lines.extend(f"- {lim}" for lim in r["limitations"])
    lines.append("")

    ds = r["dataset_summary"]
    lines += ["## Dataset summary", "", _table(
        ["metric", "value"],
        [
            ("total_records", ds["total_records"]),
            ("valid_predictions", ds["valid_predictions"]),
            ("invalid_predictions", ds["invalid_predictions"]),
            ("invalid_prediction_rate", ds["invalid_prediction_rate"]),
            ("ground_truth_count", ds["ground_truth_count"]),
            ("scene_count", ds["scene_count"]),
            ("track_count", ds["track_count"]),
            ("untracked_observations", ds["untracked_observations"]),
            ("labels", ", ".join(ds["labels"])),
            ("frame_range", ds["frame_range"]),
            ("timestamp_range", ds["timestamp_range"]),
        ]), ""]

    lines += ["## Provenance summary", "", _table(
        ["provenance", "count", "valid", "validation_class"],
        [(prov, row["count"], row["n_valid"], row["validation_class"])
         for prov, row in sorted(r["provenance_summary"].items())]), ""]

    om = r["overall_metrics"]
    lines += ["## Overall metrics", "", _table(
        ["metric", "value"],
        [
            ("n_total", om["n_total"]),
            ("n_valid", om["n_valid"]),
            ("n_invalid", om["n_invalid"]),
            ("invalid_rate", om["invalid_rate"]),
            ("provenances", ", ".join(om["provenances"])),
            ("validation_class", om["validation_class"]),
            ("MAE_m", om["mae_m"]),
            ("RMSE_m", om["rmse_m"]),
            ("median_abs_error_m", om["median_abs_error_m"]),
            ("max_abs_error_m", om["max_abs_error_m"]),
            ("median_relative_error", om["median_relative_error"]),
            ("p95_abs_error_m", om["p95_abs_error_m"]),
        ]), ""]

    lines += ["## Distance bins (ANALYSIS BINS - not safety limits)", "",
              _table(
                  ["bin_m", "samples", "valid", "invalid_rate",
                   "MAE_m", "median_abs_m", "p95_abs_m"],
                  [(b["bin_m"], b["sample_count"], b["valid_count"],
                    b["invalid_rate"], b["mae_m"],
                    b["median_abs_error_m"], b["p95_abs_error_m"])
                   for b in r["distance_bins"]]), ""]

    ed = r["error_distribution"]
    lines += ["## Error distribution", "",
              f"n_valid: {ed['n_valid']}", ""]
    lines += ["Absolute error buckets (m):", "", _table(
        ["bin_m", "count"],
        [(b["bin"], b["count"]) for b in ed["absolute_error_buckets_m"]]), ""]
    lines += ["Relative error buckets:", "", _table(
        ["bin", "count"],
        [(b["bin"], b["count"]) for b in ed["relative_error_buckets"]]), ""]

    if r.get("per_label_metrics"):
        lines += ["## Per-label metrics", "", _table(
            ["label", "samples", "valid", "invalid", "MAE_m", "RMSE_m",
             "p95_abs_m", "note"],
            [(row["name"], row["sample_count"], row["valid_count"],
              row["invalid_count"], (row["metrics"] or {}).get("mae_m"),
              (row["metrics"] or {}).get("rmse_m"),
              (row["metrics"] or {}).get("p95_abs_error_m"),
              row["note"] or "")
             for row in r["per_label_metrics"]]), ""]

    if r.get("per_scene_metrics"):
        lines += ["## Per-scene metrics", "", _table(
            ["scene", "samples", "valid", "invalid", "MAE_m", "p95_abs_m",
             "labels"],
            [(row["name"], row["sample_count"], row["valid_count"],
              row["invalid_count"], (row["metrics"] or {}).get("mae_m"),
              (row["metrics"] or {}).get("p95_abs_error_m"),
              ", ".join(row.get("labels", [])))
             for row in r["per_scene_metrics"]]), ""]

    nav = r.get("navigation_summary")
    if nav:
        lines += ["## Navigation summary (descriptive)", "", _table(
            ["metric", "value"],
            [(k, v) for k, v in sorted(nav.items())
             if k != "action_transitions"]), ""]
        if nav.get("action_transitions"):
            lines += ["Action transitions:", "", _table(
                ["transition", "count"],
                sorted(nav["action_transitions"].items())), ""]

    risk = r.get("risk_summary")
    if risk:
        lines += ["## Risk summary (descriptive)", "", _table(
            ["metric", "value"],
            [(k, v) for k, v in sorted(risk.items())
             if k != "risk_transitions"]), ""]
        if risk.get("risk_transitions"):
            lines += ["Risk transitions:", "", _table(
                ["transition", "count"],
                sorted(risk["risk_transitions"].items())), ""]

    temp = r.get("temporal_summary")
    if temp:
        lines += ["## Temporal summary (descriptive)", "", _table(
            ["metric", "value"],
            sorted(temp.items())), ""]

    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------
# File writers
# ------------------------------------------------------------------

def write_json_report(path: Union[str, Path], report: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


def write_markdown_report(path: Union[str, Path], report: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_markdown(report), encoding="utf-8")


def _cs(value) -> str:
    return "" if value is None else f"{value:.6f}"


def csv_summary_rows(report: dict) -> List[List[str]]:
    """Flat CSV rows: distance bins + per-label accuracy summaries."""
    rows = [["section", "name", "sample_count", "valid_count",
             "invalid_rate", "mae_m", "p95_abs_error_m"]]
    for b in report["distance_bins"]:
        rows.append([
            "distance_bin", b["bin_m"], str(b["sample_count"]),
            str(b["valid_count"]), _cs(b["invalid_rate"]),
            _cs(b["mae_m"]), _cs(b["p95_abs_error_m"]),
        ])
    for row in report["per_label_metrics"]:
        m = row["metrics"] or {}
        rows.append([
            "label", row["name"], str(row["sample_count"]),
            str(row["valid_count"]), _cs(m.get("invalid_rate")),
            _cs(m.get("mae_m")), _cs(m.get("p95_abs_error_m")),
        ])
    return rows


def write_csv_summary(path: Union[str, Path], report: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(csv_summary_rows(report))
