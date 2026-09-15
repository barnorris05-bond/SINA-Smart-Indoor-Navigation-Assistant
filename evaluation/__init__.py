"""
evaluation/__init__.py

Offline measurement & evaluation infrastructure (hardware-parked phase).

Public API:

    MeasurementRecord   - one distance observation (provenance-honest)
    SCHEMA_VERSION      - record schema version
    evaluate / EvaluationSummary / validation_class - accuracy metrics
    check_overlap / assert_disjoint / OverlapReport - calibration vs
                        validation separation
    write_jsonl / read_jsonl / write_summary_json - file I/O
    ScriptedDepthProvider - deterministic failure-injection provider
    ReplayRunner / ReplayConfig / ScriptedFrame / ReplayResult - offline
                        replay of the real decision pipeline
    MeasurementRecorder - live capture helper (records + event log)
    EventLogger         - structured per-frame JSONL trace

Real OAK-D hardware validation remains BLOCKED; nothing here claims
MEASURED accuracy.
"""

from evaluation.record import MeasurementRecord, SCHEMA_VERSION
from evaluation.metrics import (
    EvaluationSummary,
    OverlapReport,
    assert_disjoint,
    check_overlap,
    evaluate,
    is_valid_record,
    validation_class,
)
from evaluation.report import read_jsonl, write_jsonl, write_summary_json
from evaluation.injection import ScriptedDepthProvider, ScriptSpec
from evaluation.event_log import EventLogger
from evaluation.recorder import MeasurementRecorder
from evaluation.replay import (
    ReplayConfig,
    ReplayResult,
    ReplayRunner,
    ScriptedFrame,
)

# Offline analysis & reporting layer
from evaluation.analysis import (
    DatasetError,
    DatasetSummary,
    InvalidEntry,
    ValidatedDataset,
    dataset_summary,
    distance_bin_analysis,
    error_distribution,
    filter_label,
    filter_provenance,
    filter_scene,
    filter_track,
    group_metrics,
    label_analysis,
    load_jsonl_records,
    metrics_for,
    provenance_metrics,
    scene_analysis,
    validate_records,
)
from evaluation.navigation_analysis import (
    compare_datasets,
    navigation_stats,
    risk_stats,
    temporal_stats,
)
from evaluation.reporting import (
    analyze_dataset,
    build_report,
    evaluate_pair,
    load_jsonl_events,
    render_markdown,
    write_csv_summary,
    write_json_report,
    write_markdown_report,
)

__all__ = [
    "MeasurementRecord",
    "SCHEMA_VERSION",
    "EvaluationSummary",
    "OverlapReport",
    "assert_disjoint",
    "check_overlap",
    "evaluate",
    "is_valid_record",
    "validation_class",
    "write_jsonl",
    "read_jsonl",
    "write_summary_json",
    "ScriptedDepthProvider",
    "ScriptSpec",
    "EventLogger",
    "MeasurementRecorder",
    "ReplayConfig",
    "ReplayResult",
    "ReplayRunner",
    "ScriptedFrame",
    # analysis & reporting
    "DatasetError",
    "DatasetSummary",
    "InvalidEntry",
    "ValidatedDataset",
    "dataset_summary",
    "distance_bin_analysis",
    "error_distribution",
    "filter_label",
    "filter_provenance",
    "filter_scene",
    "filter_track",
    "group_metrics",
    "label_analysis",
    "load_jsonl_records",
    "metrics_for",
    "provenance_metrics",
    "scene_analysis",
    "validate_records",
    "compare_datasets",
    "navigation_stats",
    "risk_stats",
    "temporal_stats",
    "analyze_dataset",
    "build_report",
    "evaluate_pair",
    "load_jsonl_events",
    "render_markdown",
    "write_csv_summary",
    "write_json_report",
    "write_markdown_report",
    "measurement_quality_summary",
]
