"""
calibration/diagnostics.py

REPEATABILITY AND OUTLIER ANALYSIS (Phase 10 preparation).

Descriptive only — quantifies variability and highlights large errors;
it never deletes data and never claims statistical confidence:

- repeatability_analysis(): per-distance-group descriptive statistics
  (§18) for repeated observations at the same nominal target. Groups
  with fewer samples than min_samples are reported with
  insufficient=True and NO statistics — tiny samples are not presented
  as robust conclusions.

- outlier_report(): descriptive error distribution, largest errors,
  error-by-distance and error-by-label (§17).

- filter_outliers(): OPTIONAL filtering that is ALWAYS auditable —
  returns the kept records plus a FilterResult stating rule, count
  removed, and before/after MAE. Never silently applied: the report
  must state what was removed. Outliers are NEVER deleted from the raw
  dataset; filtering produces a derived view only.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluation.metrics import evaluate, is_valid_record
from evaluation.record import MeasurementRecord


# --------------------------------------------------------------
# Repeatability (§18)
# --------------------------------------------------------------

@dataclass(frozen=True)
class RepeatabilityGroup:
    """Descriptive variability of repeated observations at one target."""

    group_key: str              # e.g. "2.00 m"
    n_observations: int
    n_valid: int
    insufficient: bool          # True = below min_samples; no statistics
    mean_m: Optional[float] = None
    median_m: Optional[float] = None
    std_m: Optional[float] = None
    min_m: Optional[float] = None
    max_m: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "group_key": self.group_key,
            "n_observations": self.n_observations,
            "n_valid": self.n_valid,
            "insufficient": self.insufficient,
            "mean_m": self.mean_m,
            "median_m": self.median_m,
            "std_m": self.std_m,
            "min_m": self.min_m,
            "max_m": self.max_m,
        }


def repeatability_analysis(
    records: Sequence[MeasurementRecord],
    min_samples: int = 3,
    group_by: str = "ground_truth_distance_m",
) -> List[RepeatabilityGroup]:
    """
    Descriptive spread of valid predicted distances per group.

    group_by: "ground_truth_distance_m" (default), "label", or
    "scene_id". Groups are formed over ALL observations with a value
    for that key (ground-truth groups use positive truths only);
    statistics use valid records only; groups below min_samples report
    insufficient=True with no statistics.

    min_samples is a REPORTING threshold (how much data is needed
    before a spread number is shown), not a safety or acceptance
    threshold.
    """
    if min_samples < 1:
        raise ValueError(f"min_samples must be >= 1, got {min_samples}")
    if group_by not in ("ground_truth_distance_m", "label", "scene_id"):
        raise ValueError(f"unsupported group_by {group_by!r}")

    groups: Dict[str, List[MeasurementRecord]] = {}
    for r in records:
        if group_by == "ground_truth_distance_m":
            if r.ground_truth_distance_m is None or r.ground_truth_distance_m <= 0:
                continue
            key = f"{r.ground_truth_distance_m:.2f} m"
        elif group_by == "label":
            key = r.label
        else:
            key = r.scene_id
        groups.setdefault(key, []).append(r)

    out: List[RepeatabilityGroup] = []
    for key in sorted(groups):
        rows = groups[key]
        values = [
            r.predicted_distance_m for r in rows
            if is_valid_record(r) and r.predicted_distance_m is not None
        ]
        if len(values) < min_samples:
            out.append(RepeatabilityGroup(
                group_key=key, n_observations=len(rows), n_valid=len(values),
                insufficient=True,
            ))
            continue
        arr = np.array(values, dtype=float)
        out.append(RepeatabilityGroup(
            group_key=key,
            n_observations=len(rows),
            n_valid=len(values),
            insufficient=False,
            mean_m=float(np.mean(arr)),
            median_m=float(np.median(arr)),
            std_m=float(np.std(arr, ddof=1)) if len(values) > 1 else 0.0,
            min_m=float(np.min(arr)),
            max_m=float(np.max(arr)),
        ))
    return out


# --------------------------------------------------------------
# Outliers (§17)
# --------------------------------------------------------------

@dataclass(frozen=True)
class OutlierEntry:
    """One observation with a notable error (descriptive listing)."""

    scene_id: str
    frame_id: int
    track_id: Optional[int]
    label: str
    ground_truth_m: float
    predicted_m: float
    abs_error_m: float
    rel_error: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "track_id": self.track_id,
            "label": self.label,
            "ground_truth_m": self.ground_truth_m,
            "predicted_m": self.predicted_m,
            "abs_error_m": self.abs_error_m,
            "rel_error": self.rel_error,
        }


def outlier_report(
    records: Sequence[MeasurementRecord],
    top_n: int = 10,
) -> Dict[str, object]:
    """
    Descriptive outlier analysis over VALID records (§17).

    Returns absolute/relative error distributions, the largest errors,
    error-by-distance-bin means, and error-by-label means. NO data is
    removed or modified here; this is a listing, not a filter.
    """
    valid = [r for r in records if is_valid_record(r)]
    if not valid:
        return {
            "n_valid": 0,
            "abs_error_distribution": None,
            "rel_error_distribution": None,
            "largest_absolute_errors": [],
            "abs_error_by_distance_bin": {},
            "abs_error_by_label": {},
        }

    pred = np.array([r.predicted_distance_m for r in valid], dtype=float)
    gt = np.array([r.ground_truth_distance_m for r in valid], dtype=float)
    abs_err = np.abs(pred - gt)
    rel_err = abs_err / gt

    order = np.argsort(abs_err)[::-1][:max(0, int(top_n))]
    largest = [
        OutlierEntry(
            scene_id=valid[i].scene_id,
            frame_id=valid[i].frame_id,
            track_id=valid[i].track_id,
            label=valid[i].label,
            ground_truth_m=float(gt[i]),
            predicted_m=float(pred[i]),
            abs_error_m=float(abs_err[i]),
            rel_error=float(rel_err[i]),
        ).to_dict()
        for i in order
    ]

    by_bin: Dict[str, List[float]] = {}
    by_label: Dict[str, List[float]] = {}
    for i in range(len(valid)):
        bin_key = _distance_bin(gt[i])
        by_bin.setdefault(bin_key, []).append(float(abs_err[i]))
        by_label.setdefault(valid[i].label, []).append(float(abs_err[i]))

    return {
        "n_valid": len(valid),
        "abs_error_distribution": _distribution(abs_err),
        "rel_error_distribution": _distribution(rel_err),
        "largest_absolute_errors": largest,
        "abs_error_by_distance_bin": {
            k: {"mean_abs_error_m": float(np.mean(v)), "n": len(v)}
            for k, v in sorted(by_bin.items())
        },
        "abs_error_by_label": {
            k: {"mean_abs_error_m": float(np.mean(v)), "n": len(v)}
            for k, v in sorted(by_label.items())
        },
    }


def _distance_bin(gt_m: float) -> str:
    """ANALYSIS BIN (descriptive category, not a safety threshold)."""
    if gt_m < 1.0:
        return "0-1 m"
    if gt_m < 2.0:
        return "1-2 m"
    if gt_m < 3.0:
        return "2-3 m"
    if gt_m < 5.0:
        return "3-5 m"
    return "5+ m"


def _distribution(values: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50.0)),
        "p90": float(np.percentile(values, 90.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(np.max(values)),
    }


# --------------------------------------------------------------
# Auditable filtering (§17: report before/after, never silent)
# --------------------------------------------------------------

@dataclass(frozen=True)
class FilterResult:
    """Outcome of an explicit outlier-filtering step."""

    rule: str                       # human-readable rule description
    n_before: int
    n_removed: int
    n_after: int
    mae_before_m: Optional[float]   # over valid records
    mae_after_m: Optional[float]
    kept: Tuple[MeasurementRecord, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "rule": self.rule,
            "n_before": self.n_before,
            "n_removed": self.n_removed,
            "n_after": self.n_after,
            "mae_before_m": self.mae_before_m,
            "mae_after_m": self.mae_after_m,
        }


def filter_outliers(
    records: Sequence[MeasurementRecord],
    max_abs_error_m: float,
) -> FilterResult:
    """
    Keep only VALID records whose absolute error <= max_abs_error_m.

    The result is a DERIVED VIEW; callers must report it alongside the
    unfiltered baseline (FilterResult carries the rule, the count
    removed, and before/after MAE so nothing can be hidden). The raw
    dataset is never mutated.
    """
    if max_abs_error_m <= 0:
        raise ValueError(
            f"max_abs_error_m must be > 0, got {max_abs_error_m}"
        )
    before = evaluate(records).mae_m
    kept = tuple(
        r for r in records
        if is_valid_record(r)
        and r.ground_truth_distance_m is not None
        and r.predicted_distance_m is not None
        and abs(r.predicted_distance_m - r.ground_truth_distance_m)
        <= max_abs_error_m
    )
    after = evaluate(list(kept)).mae_m
    return FilterResult(
        rule=f"absolute error <= {max_abs_error_m} m",
        n_before=len(records),
        n_removed=len(records) - len(kept),
        n_after=len(kept),
        mae_before_m=before,
        mae_after_m=after,
        kept=kept,
    )
