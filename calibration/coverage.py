"""
calibration/coverage.py

COVERAGE AND AVAILABILITY ANALYSIS (Phase 10 preparation).

Descriptive only. Quantifies WHAT the experiment data covers and how
often the depth path produced a usable value:

- coverage_report(): ground-truth distance coverage (min/median/max,
  unique values), plus scene / label / target / session dimensions.
- availability_breakdown(): valid vs invalid observation rate, broken
  down by distance / label / scene / provenance (milestone §16).

"Valid" reuses evaluation.metrics.is_valid_record verbatim — a system
that is accurate only when it returns a value still has poor practical
behavior if the invalid rate is high, so availability is reported
alongside accuracy, never folded into it.

Nothing here is a safety metric, and requested target distances are
kept SEPARATE from actual ground truth (§15): the plan's nominal value
and the operator's measured truth are different quantities.
"""

from typing import Dict, List, Optional, Sequence

from evaluation.metrics import is_valid_record
from evaluation.record import MeasurementRecord


def coverage_report(
    records: Sequence[MeasurementRecord],
    requested_targets_m: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """
    Ground-truth and dimensional coverage of a dataset.

    requested_targets_m optionally lists the plan's NOMINAL targets so
    the report can show which planned distances were actually hit.
    Requested targets and observed ground truth are reported
    separately — nominal placement is not ground truth (§15).
    """
    truths = sorted(
        r.ground_truth_distance_m for r in records
        if r.ground_truth_distance_m is not None and r.ground_truth_distance_m > 0
    )

    if truths:
        n = len(truths)
        median = truths[n // 2] if n % 2 else (truths[n // 2 - 1] + truths[n // 2]) / 2
        distance_coverage: Optional[Dict[str, object]] = {
            "n_with_ground_truth": n,
            "min_m": truths[0],
            "median_m": median,
            "max_m": truths[-1],
            "unique_values": len(set(truths)),
        }
    else:
        distance_coverage = None

    report: Dict[str, object] = {
        "n_total": len(records),
        "distance_coverage": distance_coverage,
        "scenes": sorted({r.scene_id for r in records}),
        "labels": sorted({r.label for r in records}),
        "track_ids": sorted({r.track_id for r in records if r.track_id is not None}),
        "provenances": sorted({r.distance_provenance for r in records}),
    }
    if requested_targets_m is not None:
        # DISTINCT quantities: plan vs observation (§15).
        report["requested_targets_m"] = sorted(set(requested_targets_m))
        report["targets_without_ground_truth"] = (
            None if not truths else sorted(
                t for t in set(requested_targets_m)
                if not any(abs(t - g) <= 0.05 for g in truths)
            )
        )
    return report


def availability_breakdown(
    records: Sequence[MeasurementRecord],
) -> Dict[str, Dict[str, float]]:
    """
    Valid/invalid observation rate per dimension (§16).

    Returns {"overall": {...}, "by_distance": {...}, "by_label": {...},
    "by_scene": {...}, "by_provenance": {...}} where each inner dict maps
    a bucket key to {"n": int, "valid": int, "invalid": int,
    "invalid_rate": float}.

    Bucket keys:
      overall      - the constant "ALL"
      by_distance  - ground-truth value formatted "2.00 m" (records
                     without ground truth group under "no_ground_truth")
      by_label     - object label
      by_scene     - scene_id
      by_provenance- provenance vocabulary string
    """
    def bucket(rows: List[MeasurementRecord]) -> Dict[str, float]:
        valid = sum(1 for r in rows if is_valid_record(r))
        n = len(rows)
        return {
            "n": n,
            "valid": valid,
            "invalid": n - valid,
            "invalid_rate": ((n - valid) / n) if n else 0.0,
        }

    by_distance: Dict[str, Dict[str, float]] = {}
    by_label: Dict[str, Dict[str, float]] = {}
    by_scene: Dict[str, Dict[str, float]] = {}
    by_provenance: Dict[str, Dict[str, float]] = {}

    for r in records:
        key = (
            "no_ground_truth" if r.ground_truth_distance_m is None
            else f"{r.ground_truth_distance_m:.2f} m"
        )
        by_distance.setdefault(key, []).append(r)
        by_label.setdefault(r.label, []).append(r)
        by_scene.setdefault(r.scene_id, []).append(r)
        by_provenance.setdefault(r.distance_provenance, []).append(r)

    return {
        "overall": {"ALL": bucket(list(records))},
        "by_distance": {k: bucket(v) for k, v in sorted(by_distance.items())},
        "by_label": {k: bucket(v) for k, v in sorted(by_label.items())},
        "by_scene": {k: bucket(v) for k, v in sorted(by_scene.items())},
        "by_provenance": {k: bucket(v) for k, v in sorted(by_provenance.items())},
    }
