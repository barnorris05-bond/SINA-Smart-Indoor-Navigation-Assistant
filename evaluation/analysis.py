"""
evaluation/analysis.py

OFFLINE ANALYSIS LAYER over the existing evaluation package.

Pipeline (software-only; works on SIMULATED data today and on MEASURED
data later without architectural change):

    recorded JSONL -> load -> validate -> filter -> evaluate -> summarize

Design rules:

- REUSE, do not duplicate: MeasurementRecord is the only record model;
  accuracy metrics come exclusively from evaluation/metrics.py. This
  module adds loading, validation triage, dataset summary, filtering
  and grouping - never new metric formulas.
- PROVENANCE is never silently combined: every summary lists provenance
  counts, every metric group is splittable by provenance, and the
  validation classification from metrics.validation_class() travels
  with every result.
- BAD DATA never crashes analysis: malformed lines, unknown
  provenance, missing ground truth and non-positive values are triaged
  into explicit invalid buckets with reasons (see validate_records).
- DETERMINISTIC: no wall-clock, no randomness; all groupings are
  sorted; the same input always produces the same output.

ANALYSIS BINS: the distance/error bin edges below are ANALYSIS
categories for describing datasets. They are NOT safety limits and
have no relationship to navigation thresholds (config/navigation_*).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from evaluation.record import MeasurementRecord, VALID_PROVENANCES
from evaluation.metrics import (
    EvaluationSummary,
    evaluate,
    is_valid_record,
    validation_class,
)

# Canonical provenance display order (project vocabulary).
PROVENANCE_ORDER = ("MEASURED", "SIMULATED", "UNAVAILABLE", "STALE")

# ------------------------------------------------------------------
# ANALYSIS BINS (descriptive categories - NOT safety limits)
# ------------------------------------------------------------------

DISTANCE_BINS_M = (1.0, 2.0, 3.0, 4.0, 5.0)   # ground-truth meters
ERROR_ABS_BINS_M = (0.05, 0.10, 0.25, 0.50, 1.0)
ERROR_REL_BINS = (0.05, 0.10, 0.25, 0.50)


class DatasetError(ValueError):
    """Raised for unusable datasets with an actionable message."""


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_jsonl_records(path: Union[str, Path]) -> List[MeasurementRecord]:
    """
    Load a MeasurementRecord JSONL file.

    Blank lines are skipped. A structurally malformed line raises
    DatasetError naming the line number - the file as a whole is
    rejected so silent partial reads can never bias a report.
    """
    records: List[MeasurementRecord] = []
    p = Path(path)
    if not p.is_file():
        raise DatasetError(f"records file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(MeasurementRecord.parse_json(line))
            except Exception as exc:
                raise DatasetError(
                    f"{p}: malformed record at line {lineno}: {exc}"
                ) from exc
    return records


# ------------------------------------------------------------------
# Validation triage
# ------------------------------------------------------------------

@dataclass(frozen=True)
class InvalidEntry:
    """One record excluded from accuracy metrics, with the reason why."""

    index: int
    reason: str      # see validate_records docstring
    detail: str      # identity key / value for the report


@dataclass(frozen=True)
class ValidatedDataset:
    """Result of triaging a record list before evaluation."""

    records: List[MeasurementRecord]   # all parsed records (unchanged)
    invalid: List[InvalidEntry]        # excluded from accuracy metrics

    @property
    def valid_for_accuracy(self) -> List[MeasurementRecord]:
        return [
            r for i, r in enumerate(self.records)
            if i not in {e.index for e in self.invalid}
        ]


def _triage_one(rec: MeasurementRecord) -> Optional[str]:
    """Return an invalidity reason for rec, or None if accuracy-valid."""
    p, prov, gt = (
        rec.predicted_distance_m,
        rec.distance_provenance,
        rec.ground_truth_distance_m,
    )
    if gt is None:
        return "missing_ground_truth"
    if gt <= 0:
        return "non_positive_ground_truth"
    if prov not in VALID_PROVENANCES:
        return "unknown_provenance"
    if p is None:
        if prov == "STALE":
            return "stale_prediction"
        if prov == "UNAVAILABLE":
            return "unavailable_prediction"
        return "missing_prediction"
    if p <= 0:
        return "non_positive_prediction"
    if prov not in ("MEASURED", "SIMULATED"):
        # Numeric prediction carried on a non-usable provenance:
        # never allowed into accuracy metrics (no silent promotion).
        return "non_usable_provenance_with_value"
    return None


def validate_records(
    records: Sequence[MeasurementRecord],
) -> ValidatedDataset:
    """
    Triage records for accuracy evaluation.

    A record is VALID for accuracy iff metrics.is_valid_record() holds
    (usable provenance, positive prediction AND positive ground truth).
    Everything else is listed in .invalid with one explicit reason:

        missing_ground_truth | non_positive_ground_truth |
        unknown_provenance   | stale_prediction |
        unavailable_prediction | missing_prediction |
        non_positive_prediction | non_usable_provenance_with_value

    This never mutates or drops records from .records.
    """
    invalid: List[InvalidEntry] = []
    for i, rec in enumerate(records):
        reason = _triage_one(rec)
        if reason is not None:
            invalid.append(InvalidEntry(
                index=i, reason=reason, detail=str(rec.identity_key())))
        elif not is_valid_record(rec):
            # Defense in depth: the triage and the metric gate must
            # agree; if not, exclude conservatively.
            invalid.append(InvalidEntry(
                index=i, reason="failed_validity_check",
                detail=str(rec.identity_key())))
    return ValidatedDataset(records=list(records), invalid=invalid)


# ------------------------------------------------------------------
# Dataset summary
# ------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSummary:
    """Descriptive dataset summary (no accuracy inference)."""

    total_records: int
    valid_predictions: int          # usable provenance + numeric > 0
    invalid_predictions: int        # None / stale / unavailable / <= 0
    invalid_prediction_rate: float
    provenance_counts: Dict[str, int]
    ground_truth_count: int
    scene_count: int
    scene_ids: Tuple[str, ...]
    track_count: int                # distinct non-null track ids
    untracked_observations: int     # records with track_id None
    labels: Tuple[str, ...]
    frame_range: Optional[Tuple[int, int]]
    timestamp_range: Optional[Tuple[float, float]]

    def to_dict(self) -> dict:
        return {
            "total_records": self.total_records,
            "valid_predictions": self.valid_predictions,
            "invalid_predictions": self.invalid_predictions,
            "invalid_prediction_rate": self.invalid_prediction_rate,
            "provenance_counts": dict(self.provenance_counts),
            "ground_truth_count": self.ground_truth_count,
            "scene_count": self.scene_count,
            "scene_ids": list(self.scene_ids),
            "track_count": self.track_count,
            "untracked_observations": self.untracked_observations,
            "labels": list(self.labels),
            "frame_range": (
                None if self.frame_range is None
                else list(self.frame_range)
            ),
            "timestamp_range": (
                None if self.timestamp_range is None
                else list(self.timestamp_range)
            ),
        }


def dataset_summary(records: Sequence[MeasurementRecord]) -> DatasetSummary:
    """Describe a dataset without interpreting it (missing metadata is
    reported as absent, never inferred)."""
    provenance_counts = {p: 0 for p in PROVENANCE_ORDER}
    for rec in records:
        provenance_counts.setdefault(rec.distance_provenance, 0)
        provenance_counts[rec.distance_provenance] += 1

    frames = [r.frame_id for r in records]
    stamps = [r.timestamp for r in records]
    track_ids = {r.track_id for r in records if r.track_id is not None}
    valid_predictions = sum(
        1 for r in records
        if r.predicted_distance_m is not None
        and r.predicted_distance_m > 0
        and r.distance_provenance in ("MEASURED", "SIMULATED")
    )
    return DatasetSummary(
        total_records=len(records),
        valid_predictions=valid_predictions,
        invalid_predictions=len(records) - valid_predictions,
        invalid_prediction_rate=(
            (len(records) - valid_predictions) / len(records)
            if records else 0.0
        ),
        provenance_counts=provenance_counts,
        ground_truth_count=sum(
            1 for r in records if r.ground_truth_distance_m is not None),
        scene_count=len({r.scene_id for r in records}),
        scene_ids=tuple(sorted({r.scene_id for r in records})),
        track_count=len(track_ids),
        untracked_observations=sum(
            1 for r in records if r.track_id is None),
        labels=tuple(sorted({r.label for r in records})),
        frame_range=(
            (min(frames), max(frames)) if frames else None),
        timestamp_range=(
            (min(stamps), max(stamps)) if stamps else None),
    )


# ------------------------------------------------------------------
# Filters (non-mutating; return new lists)
# ------------------------------------------------------------------

def _as_iter(value: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def filter_provenance(
    records: Sequence[MeasurementRecord],
    provenances: Union[str, Iterable[str]],
) -> List[MeasurementRecord]:
    """Keep records whose provenance is in the given set."""
    wanted = set(_as_iter(provenances))
    unknown = wanted - set(VALID_PROVENANCES)
    if unknown:
        raise DatasetError(
            f"unknown provenance filter {sorted(unknown)}; "
            f"valid: {list(VALID_PROVENANCES)}")
    return [r for r in records if r.distance_provenance in wanted]


def filter_scene(
    records: Sequence[MeasurementRecord],
    scene_ids: Union[str, Iterable[str]],
) -> List[MeasurementRecord]:
    wanted = set(_as_iter(scene_ids))
    return [r for r in records if r.scene_id in wanted]


def filter_label(
    records: Sequence[MeasurementRecord],
    labels: Union[str, Iterable[str]],
) -> List[MeasurementRecord]:
    wanted = set(_as_iter(labels))
    return [r for r in records if r.label in wanted]


def filter_track(
    records: Sequence[MeasurementRecord],
    track_ids: Union[int, Iterable[int]],
) -> List[MeasurementRecord]:
    if isinstance(track_ids, int):
        track_ids = [track_ids]
    wanted = set(track_ids)
    return [r for r in records if r.track_id in wanted]


# ------------------------------------------------------------------
# Grouped metrics (thin wrappers over metrics.evaluate - no new formulas)
# ------------------------------------------------------------------

def metrics_for(records: Sequence[MeasurementRecord]) -> EvaluationSummary:
    """Overall accuracy metrics for a record list (see metrics.py)."""
    return evaluate(list(records))


def group_metrics(
    records: Sequence[MeasurementRecord],
    key_fn: Callable[[MeasurementRecord], str],
) -> Dict[str, EvaluationSummary]:
    """Metrics per group key (sorted keys; empty groups omitted)."""
    groups: Dict[str, List[MeasurementRecord]] = {}
    for rec in records:
        groups.setdefault(key_fn(rec), []).append(rec)
    return {
        key: evaluate(groups[key]) for key in sorted(groups)
    }


def provenance_metrics(
    records: Sequence[MeasurementRecord],
) -> Dict[str, EvaluationSummary]:
    """Metrics per provenance - MEASURED and SIMULATED are never mixed."""
    return group_metrics(records, lambda r: r.distance_provenance)


def label_metrics(
    records: Sequence[MeasurementRecord],
) -> Dict[str, EvaluationSummary]:
    return group_metrics(records, lambda r: r.label)


def scene_metrics(
    records: Sequence[MeasurementRecord],
) -> Dict[str, EvaluationSummary]:
    return group_metrics(records, lambda r: r.scene_id)


# ------------------------------------------------------------------
# Error / distance distributions (ANALYSIS BINS - not safety limits)
# ------------------------------------------------------------------

def error_distribution(
    records: Sequence[MeasurementRecord],
    abs_bins: Sequence[float] = ERROR_ABS_BINS_M,
    rel_bins: Sequence[float] = ERROR_REL_BINS,
) -> dict:
    """
    Error distribution over accuracy-valid records.

    Buckets are ANALYSIS BINS for describing datasets - not safety
    limits, not navigation thresholds. Absolute errors in meters;
    relative errors are |p - gt| / gt.
    """
    valid = [r for r in records if is_valid_record(r)]
    abs_errs = [abs(r.predicted_distance_m - r.ground_truth_distance_m)
                for r in valid]
    rel_errs = [e / r.ground_truth_distance_m
                for e, r in zip(abs_errs, valid)]

    def bucketize(values: Sequence[float], edges: Sequence[float]) -> List[dict]:
        labels = []
        prev = 0.0
        for hi in edges:
            labels.append(f"{prev:g}-{hi:g}")
            prev = hi
        labels.append(f">{prev:g}")
        counts = [0] * len(labels)
        for v in values:
            idx = len(edges)
            for j, hi in enumerate(edges):
                if v <= hi:
                    idx = j
                    break
            counts[idx] += 1
        return [{"bin": lab, "count": c} for lab, c in zip(labels, counts)]

    # Count by ground-truth distance range (same ANALYSIS bins as the
    # distance-bin analysis, so the two views stay comparable).
    by_distance = bucketize(
        [r.ground_truth_distance_m for r in valid], DISTANCE_BINS_M)

    return {
        "n_valid": len(valid),
        "absolute_error_buckets_m": bucketize(abs_errs, abs_bins),
        "relative_error_buckets": bucketize(rel_errs, rel_bins),
        "count_by_ground_truth_range_m": by_distance,
    }


def distance_bin_analysis(
    records: Sequence[MeasurementRecord],
    bins: Sequence[float] = DISTANCE_BINS_M,
) -> List[dict]:
    """
    Accuracy grouped by GROUND-TRUTH distance bin.

    Records without ground truth cannot be distance-binned; they are
    reported once under "no_ground_truth". Per bin:

        sample_count   - records in the bin (valid + invalid)
        valid_count    - accuracy-valid records
        invalid_rate   - 1 - valid/sample_count
        mae_m / median_abs_error_m / p95_abs_error_m over valid records
        (None when the bin has no valid records - never zero)

    Bin edges are ANALYSIS BINS - not STOP/WARNING/SAFETY distances and
    they never modify navigation.
    """
    edges = list(bins)
    labels: List[str] = []
    prev = 0.0
    for hi in edges:
        labels.append(f"{prev:g}-{hi:g}")
        prev = hi
    labels.append(f">{prev:g}")

    binned: Dict[str, List[MeasurementRecord]] = {lab: [] for lab in labels}
    no_truth: List[MeasurementRecord] = []
    for rec in records:
        gt = rec.ground_truth_distance_m
        if gt is None or gt <= 0:
            no_truth.append(rec)
            continue
        lab = labels[-1]
        for j, hi in enumerate(edges):
            if gt <= hi:
                lab = labels[j]
                break
        binned[lab].append(rec)

    out: List[dict] = []
    for lab in labels:
        group = binned[lab]
        summary = evaluate(group)
        out.append({
            "bin_m": lab,
            "sample_count": len(group),
            "valid_count": summary.n_valid,
            "invalid_rate": summary.invalid_rate,
            "mae_m": summary.mae_m,
            "median_abs_error_m": summary.median_abs_error_m,
            "p95_abs_error_m": summary.p95_abs_error_m,
        })
    out.append({
        "bin_m": "no_ground_truth",
        "sample_count": len(no_truth),
        "valid_count": 0,
        "invalid_rate": None,
        "mae_m": None,
        "median_abs_error_m": None,
        "p95_abs_error_m": None,
    })
    return out


# ------------------------------------------------------------------
# Label / scene analysis
# ------------------------------------------------------------------

def _group_rows(
    records: Sequence[MeasurementRecord],
    key_fn: Callable[[MeasurementRecord], str],
) -> List[dict]:
    groups: Dict[str, List[MeasurementRecord]] = {}
    for rec in records:
        groups.setdefault(key_fn(rec), []).append(rec)

    rows: List[dict] = []
    for key in sorted(groups):
        group = groups[key]
        summary = evaluate(group)
        provenances = {p: 0 for p in PROVENANCE_ORDER}
        for rec in group:
            provenances.setdefault(rec.distance_provenance, 0)
            provenances[rec.distance_provenance] += 1
        row = {
            "name": key,
            "sample_count": len(group),
            "valid_count": summary.n_valid,
            "invalid_count": summary.n_invalid,
            "provenance_counts": provenances,
        }
        if summary.n_valid == 0:
            row["metrics"] = None
            row["note"] = "insufficient data: no valid measurements"
        else:
            row["metrics"] = summary.to_dict()
            row["note"] = None
        rows.append(row)
    return rows


def label_analysis(records: Sequence[MeasurementRecord]) -> List[dict]:
    """Per-label descriptive analysis (see _group_rows for the shape)."""
    return _group_rows(records, lambda r: r.label)


def scene_analysis(records: Sequence[MeasurementRecord]) -> List[dict]:
    """Per-scene descriptive analysis; no unrecorded properties inferred."""
    rows = _group_rows(records, lambda r: r.scene_id)
    labels_by_scene: Dict[str, List[str]] = {}
    for rec in records:
        labels_by_scene.setdefault(rec.scene_id, [])
        if rec.label not in labels_by_scene[rec.scene_id]:
            labels_by_scene[rec.scene_id].append(rec.label)
    for row in rows:
        row["labels"] = sorted(labels_by_scene.get(row["name"], []))
    return rows
