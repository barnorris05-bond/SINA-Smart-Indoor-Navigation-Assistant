"""
tests/test_evaluation_analysis.py

OFFLINE ANALYSIS / REPORTING / CLI TESTS (hardware-free).

Covers (milestone §20):

  - loading (round-trip, blanks, malformed line -> DatasetError with
    line number, missing file)
  - validation triage (missing/zero/negative ground truth, unknown
    provenance, stale/unavailable predictions, mixed provenance)
  - dataset summary (counts, provenance counts, scenes, tracks,
    labels, frame/timestamp ranges; empty dataset)
  - provenance/scene/label/track filtering (incl. unknown-filter error)
  - overall + grouped metrics (reusing metrics.py; provenance never
    silently mixed)
  - error distributions and ANALYSIS-BIN distance bins
  - navigation / risk / temporal descriptive statistics
  - before/after DESCRIPTIVE comparison
  - report building (roles, determinism, limitations), evaluate_pair
    with enforced disjointness, markdown/CSV rendering, JSON writers
  - CLI behavior (success, JSON/MD/CSV outputs, malformed input,
    empty input)

No OAK-D, no DepthAI import, no network, no GPU.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.record import MeasurementRecord, SCHEMA_VERSION
from evaluation.analysis import (
    DatasetError,
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
    csv_summary_rows,
    evaluate_pair,
    load_jsonl_events,
    render_markdown,
    write_csv_summary,
    write_json_report,
    write_markdown_report,
)


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def make_record(**overrides) -> MeasurementRecord:
    defaults = dict(
        schema_version=SCHEMA_VERSION,
        scene_id="scene",
        frame_id=0,
        timestamp=0.0,
        track_id=None,
        label="person",
        confidence=0.9,
        bbox=(100, 100, 200, 300),
        region="CENTER",
        predicted_distance_m=2.0,
        distance_provenance="SIMULATED",
        distance_source="test",
        ground_truth_distance_m=2.0,
    )
    defaults.update(overrides)
    return MeasurementRecord(**defaults)


def _write_records(path: Path, records) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(r.to_json() + "\n" for r in records), encoding="utf-8")
    return str(path)


def _event(action, objects=None, frame=0):
    return {"type": "frame", "ts": frame / 15.0, "frame": frame,
            "action": action, "action_reason": "test", "priority": 10,
            "objects": objects if objects is not None else []}


def _person_obj(risk="HIGH", ttc=None, track=1):
    return {"track_id": track, "label": "person", "region": "CENTER",
            "confidence": 0.9, "distance": 2.0,
            "distance_provenance": "MEASURED", "motion_state": "APPROACHING",
            "closing_rate_mps": 0.5, "risk_level": risk, "risk_ttc_s": ttc}


# ==========================================================
# Loading
# ==========================================================

class TestLoading:
    def test_roundtrip_and_blank_lines(self, tmp_path):
        p = tmp_path / "r.jsonl"
        recs = [make_record(frame_id=i) for i in range(3)]
        p.write_text(
            "\n".join(r.to_json() for r in recs) + "\n\n\n", encoding="utf-8")
        loaded = load_jsonl_records(p)
        assert loaded == recs

    def test_missing_file(self, tmp_path):
        with pytest.raises(DatasetError, match="not found"):
            load_jsonl_records(tmp_path / "nope.jsonl")

    def test_malformed_line_names_lineno(self, tmp_path):
        p = tmp_path / "r.jsonl"
        p.write_text(
            make_record().to_json() + "\n{not json}\n", encoding="utf-8")
        with pytest.raises(DatasetError, match="line 2"):
            load_jsonl_records(p)

    def test_unknown_provenance_line_rejected(self, tmp_path):
        d = make_record().to_dict()
        d["distance_provenance"] = "MAGICAL"
        p = tmp_path / "r.jsonl"
        p.write_text(make_record().to_json() + "\n" + json.dumps(d) + "\n",
                     encoding="utf-8")
        with pytest.raises(DatasetError, match="line 2"):
            load_jsonl_records(p)


# ==========================================================
# Validation triage
# ==========================================================

class TestValidationTriage:
    def test_clean_dataset_has_no_invalid(self):
        v = validate_records([make_record(), make_record(frame_id=1)])
        assert v.invalid == []
        assert len(v.valid_for_accuracy) == 2

    @pytest.mark.parametrize("override,reason", [
        ({"ground_truth_distance_m": None}, "missing_ground_truth"),
        ({"ground_truth_distance_m": 0.0}, "non_positive_ground_truth"),
        ({"ground_truth_distance_m": -1.0}, "non_positive_ground_truth"),
        ({"predicted_distance_m": None,
          "distance_provenance": "UNAVAILABLE"}, "unavailable_prediction"),
        ({"predicted_distance_m": None,
          "distance_provenance": "STALE"}, "stale_prediction"),
        ({"predicted_distance_m": 0.0}, "non_positive_prediction"),
        ({"predicted_distance_m": -2.0}, "non_positive_prediction"),
        ({"predicted_distance_m": 1.0,
          "distance_provenance": "UNAVAILABLE"},
         "non_usable_provenance_with_value"),
    ])
    def test_invalid_reasons(self, override, reason):
        v = validate_records([make_record(**override)])
        assert [e.reason for e in v.invalid] == [reason]
        assert v.valid_for_accuracy == []

    def test_mixed_provenance_all_counted(self):
        recs = [
            make_record(frame_id=0, distance_provenance="SIMULATED"),
            make_record(frame_id=1, distance_provenance="MEASURED"),
            make_record(frame_id=2, predicted_distance_m=None,
                        distance_provenance="UNAVAILABLE"),
        ]
        v = validate_records(recs)
        assert v.invalid[0].reason == "unavailable_prediction"
        assert len(v.valid_for_accuracy) == 2

    def test_records_not_mutated(self):
        rec = make_record(predicted_distance_m=None,
                          distance_provenance="STALE")
        validate_records([rec])
        assert rec.predicted_distance_m is None
        assert rec.distance_provenance == "STALE"


# ==========================================================
# Dataset summary
# ==========================================================

class TestDatasetSummary:
    def test_full_summary(self):
        recs = [
            make_record(scene_id="s1", frame_id=2, track_id=1, label="person",
                        timestamp=5.0),
            make_record(scene_id="s1", frame_id=3, track_id=1, label="chair",
                        timestamp=6.0),
            make_record(scene_id="s2", frame_id=1, track_id=2, label="person",
                        timestamp=9.0,
                        predicted_distance_m=None,
                        distance_provenance="UNAVAILABLE"),
        ]
        s = dataset_summary(recs)
        assert s.total_records == 3
        assert s.valid_predictions == 2
        assert s.invalid_predictions == 1
        assert s.invalid_prediction_rate == pytest.approx(1 / 3)
        assert s.provenance_counts["SIMULATED"] == 2
        assert s.provenance_counts["UNAVAILABLE"] == 1
        assert s.ground_truth_count == 3
        assert s.scene_count == 2 and s.scene_ids == ("s1", "s2")
        assert s.track_count == 2
        assert s.untracked_observations == 0
        assert s.labels == ("chair", "person")
        assert s.frame_range == (1, 3)
        assert s.timestamp_range == (5.0, 9.0)

    def test_empty_dataset(self):
        s = dataset_summary([])
        assert s.total_records == 0
        assert s.frame_range is None
        assert s.timestamp_range is None
        assert s.invalid_prediction_rate == 0.0

    def test_to_dict_json_safe(self):
        d = dataset_summary([make_record(track_id=7)]).to_dict()
        assert d["track_count"] == 1
        assert d["frame_range"] == [0, 0]


# ==========================================================
# Filters
# ==========================================================

class TestFilters:
    RECS = [
        make_record(scene_id="s1", label="person", track_id=1,
                    distance_provenance="MEASURED"),
        make_record(scene_id="s2", label="chair", track_id=2,
                    distance_provenance="SIMULATED"),
        make_record(scene_id="s1", label="bottle", track_id=3,
                    distance_provenance="UNAVAILABLE"),
    ]

    def test_filter_provenance(self):
        out = filter_provenance(self.RECS, "MEASURED")
        assert [r.label for r in out] == ["person"]
        out2 = filter_provenance(self.RECS, ["MEASURED", "SIMULATED"])
        assert len(out2) == 2

    def test_filter_provenance_unknown_raises(self):
        with pytest.raises(DatasetError, match="unknown provenance"):
            filter_provenance(self.RECS, "MAGICAL")

    def test_filter_scene_label_track(self):
        assert [r.scene_id for r in filter_scene(self.RECS, "s1")] == \
            ["s1", "s1"]
        assert [r.label for r in filter_label(self.RECS, "chair")] == ["chair"]
        assert [r.track_id for r in filter_track(self.RECS, [1, 3])] == \
            [1, 3]

    def test_filters_do_not_mutate(self):
        before = [r.to_dict() for r in self.RECS]
        filter_scene(self.RECS, "s1")
        filter_label(self.RECS, "person")
        assert [r.to_dict() for r in self.RECS] == before


# ==========================================================
# Metrics reuse + grouping
# ==========================================================

class TestGroupedMetrics:
    def test_overall_metrics_match_metrics_module(self):
        recs = [make_record(frame_id=i, predicted_distance_m=1.5,
                            ground_truth_distance_m=1.0) for i in range(4)]
        s = metrics_for(recs)
        assert s.n_valid == 4
        assert s.mae_m == pytest.approx(0.5)

    def test_provenance_groups_never_mixed(self):
        recs = [
            make_record(frame_id=0, distance_provenance="MEASURED",
                        predicted_distance_m=2.0,
                        ground_truth_distance_m=2.0),
            make_record(frame_id=1, distance_provenance="SIMULATED",
                        predicted_distance_m=3.0,
                        ground_truth_distance_m=1.0),
        ]
        groups = provenance_metrics(recs)
        assert set(groups) == {"MEASURED", "SIMULATED"}
        assert groups["MEASURED"].mae_m == pytest.approx(0.0)
        assert groups["SIMULATED"].mae_m == pytest.approx(2.0)

    def test_group_metrics_sorted_keys(self):
        recs = [make_record(label="chair"), make_record(label="person")]
        groups = group_metrics(recs, lambda r: r.label)
        assert list(groups) == ["chair", "person"]


# ==========================================================
# Error distribution + distance bins
# ==========================================================

class TestDistributions:
    def test_error_distribution_buckets(self):
        recs = [
            make_record(predicted_distance_m=1.02,
                        ground_truth_distance_m=1.0),   # abs 0.02
            make_record(predicted_distance_m=1.30,
                        ground_truth_distance_m=1.0),   # abs 0.30
            make_record(predicted_distance_m=3.00,
                        ground_truth_distance_m=1.0),   # abs 2.00
        ]
        dist = error_distribution(recs)
        assert dist["n_valid"] == 3
        buckets = {b["bin"]: b["count"]
                   for b in dist["absolute_error_buckets_m"]}
        assert buckets["0-0.05"] == 1
        assert buckets["0.25-0.5"] == 1
        assert buckets[">1"] == 1

    def test_error_distribution_counts_by_gt_range(self):
        recs = [
            make_record(predicted_distance_m=0.5,
                        ground_truth_distance_m=0.5),
            make_record(predicted_distance_m=1.5,
                        ground_truth_distance_m=1.5),
            make_record(predicted_distance_m=9.0,
                        ground_truth_distance_m=9.0),
        ]
        dist = error_distribution(recs)
        counts = {b["bin"]: b["count"]
                  for b in dist["count_by_ground_truth_range_m"]}
        assert counts["0-1"] == 1
        assert counts["1-2"] == 1
        assert counts[">5"] == 1

    def test_distance_bins_report_all_bins(self):
        recs = [
            make_record(predicted_distance_m=1.2,
                        ground_truth_distance_m=1.0),
            make_record(predicted_distance_m=2.4,
                        ground_truth_distance_m=2.0),
            make_record(predicted_distance_m=7.0,
                        ground_truth_distance_m=7.0),
        ]
        bins = distance_bin_analysis(recs)
        labels = [b["bin_m"] for b in bins]
        assert labels == ["0-1", "1-2", "2-3", "3-4", "4-5", ">5",
                          "no_ground_truth"]
        by_bin = {b["bin_m"]: b for b in bins}
        assert by_bin["0-1"]["sample_count"] == 1
        assert by_bin["0-1"]["mae_m"] == pytest.approx(0.2)
        assert by_bin[">5"]["sample_count"] == 1
        assert by_bin[">5"]["mae_m"] == pytest.approx(0.0)
        assert by_bin["3-4"]["sample_count"] == 0

    def test_distance_bins_no_ground_truth_grouped(self):
        recs = [make_record(ground_truth_distance_m=None)]
        bins = distance_bin_analysis(recs)
        no_truth = bins[-1]
        assert no_truth["bin_m"] == "no_ground_truth"
        assert no_truth["sample_count"] == 1
        assert no_truth["mae_m"] is None   # never zero

    def test_bins_are_analysis_categories_not_safety(self):
        """The analysis layer must never read navigation thresholds:
        analysis bins are descriptive categories, not safety limits."""
        import evaluation.analysis as mod
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "config.navigation" not in src
        assert "CRITICAL_DISTANCE" not in src
        assert "WARNING_DISTANCE" not in src
        assert "FAR_DISTANCE" not in src


# ==========================================================
# Label / scene analysis
# ==========================================================

class TestLabelSceneAnalysis:
    def test_label_rows(self):
        recs = [
            make_record(label="person", predicted_distance_m=2.0,
                        ground_truth_distance_m=2.0),
            make_record(label="person", predicted_distance_m=2.2,
                        ground_truth_distance_m=2.0),
            make_record(label="chair", predicted_distance_m=None,
                        distance_provenance="UNAVAILABLE"),
        ]
        rows = {r["name"]: r for r in label_analysis(recs)}
        assert rows["person"]["sample_count"] == 2
        assert rows["person"]["metrics"]["mae_m"] == pytest.approx(0.1)
        assert rows["chair"]["metrics"] is None
        assert "insufficient" in rows["chair"]["note"]

    def test_scene_rows_carry_labels(self):
        recs = [
            make_record(scene_id="s1", label="person"),
            make_record(scene_id="s1", label="chair"),
            make_record(scene_id="s2", label="person"),
        ]
        rows = {r["name"]: r for r in scene_analysis(recs)}
        assert rows["s1"]["labels"] == ["chair", "person"]
        assert rows["s2"]["labels"] == ["person"]

    def test_scene_with_no_valid_measurements(self):
        recs = [make_record(scene_id="s1", predicted_distance_m=None,
                            distance_provenance="UNAVAILABLE")]
        rows = scene_analysis(recs)
        assert rows[0]["metrics"] is None
        assert rows[0]["note"] is not None


# ==========================================================
# Navigation / risk / temporal statistics
# ==========================================================

class TestNavigationStats:
    def test_counts_and_transitions(self):
        events = [_event("CONTINUE", frame=0), _event("SLOW_DOWN", frame=1),
                  _event("SLOW_DOWN", frame=2), _event("STOP", frame=3),
                  _event("STOP", frame=4), _event("STOP", frame=5)]
        s = navigation_stats(events)
        assert s["frames"] == 6
        assert s["action_counts"]["CONTINUE"] == 1
        assert s["action_counts"]["SLOW_DOWN"] == 2
        assert s["action_counts"]["STOP"] == 3
        # Two genuine A->B transitions: CONTINUE->SLOW_DOWN, SLOW_DOWN->STOP
        assert s["action_change_count"] == 2
        # Consistency invariant: changes == sum of transition counts
        assert s["action_change_count"] == sum(
            s["action_transitions"].values())
        assert s["stop_frames"] == 3
        assert s["stop_run_count"] == 1
        assert s["longest_stop_run_frames"] == 3
        assert s["action_transitions"]["CONTINUE->SLOW_DOWN"] == 1
        assert s["action_transitions"]["SLOW_DOWN->STOP"] == 1

    def test_escalation_counting(self):
        events = [_event("CONTINUE"), _event("SLOW_DOWN"), _event("STOP"),
                  _event("SLOW_DOWN"), _event("CONTINUE")]
        s = navigation_stats(events)
        assert s["escalation_events"] == 2   # CONTINUE->SLOW_DOWN, SLOW->STOP

    def test_direction_flips(self):
        events = [_event("MOVE_LEFT"), _event("MOVE_RIGHT"),
                  _event("MOVE_LEFT"), _event("MOVE_RIGHT")]
        t = temporal_stats(events)
        assert t["direction_changes"] == 3

    def test_stable_run_detection(self):
        events = [_event("CONTINUE")] * 5 + [_event("STOP")] * 2
        t = temporal_stats(events)
        assert t["longest_stable_action_run"] == 5
        assert t["stop_run_count"] == 1

    def test_empty_events(self):
        assert navigation_stats([])["frames"] == 0
        assert temporal_stats([])["frames"] == 0


class TestRiskStats:
    def test_counts_ttc_and_transitions(self):
        events = [
            _event("STOP", [_person_obj(risk="HIGH", ttc=2.5)]),
            _event("STOP", [_person_obj(risk="CRITICAL", ttc=0.9)]),
            _event("STOP", [_person_obj(risk="CRITICAL", ttc=0.5)]),
        ]
        s = risk_stats(events)
        assert s["risk_level_counts"]["HIGH"] == 1
        assert s["risk_level_counts"]["CRITICAL"] == 2
        assert s["risk_transitions"]["HIGH->CRITICAL"] == 1
        assert s["ttc_observations"] == 3
        assert s["ttc_min_s"] == pytest.approx(0.5)
        assert s["ttc_median_s"] == pytest.approx(0.9)

    def test_no_risk_data(self):
        s = risk_stats([_event("CONTINUE", [])])
        assert s["risk_level_counts"]["LOW"] == 0
        assert s["ttc_observations"] == 0
        assert s["ttc_min_s"] is None

    def test_negative_ttc_never_counted(self):
        s = risk_stats([_event("STOP", [_person_obj(ttc=-1.0)])])
        assert s["ttc_observations"] == 0


# ==========================================================
# Before/after comparison (descriptive)
# ==========================================================

class TestCompareDatasets:
    def test_descriptive_comparison_labels_and_differences(self):
        baseline = [_event("CONTINUE")] * 6
        treatment = [_event("CONTINUE"), _event("STOP"), _event("STOP"),
                     _event("STOP"), _event("STOP"), _event("STOP")]
        cmp = compare_datasets(baseline, treatment,
                               baseline_name="baseline",
                               treatment_name="temporal")
        assert "DESCRIPTIVE" in cmp["comparison_type"]
        assert cmp["datasets"]["baseline"]["navigation"]["frames"] == 6
        d = cmp["differences"]
        assert d["action_changes"] == 1
        assert d["stop_frames"] == 5

    def test_no_causal_language_in_output(self):
        cmp = compare_datasets([_event("STOP")] * 3, [_event("STOP")] * 3)
        text = json.dumps(cmp).lower()
        assert "proven" not in text
        assert "improve" not in text


# ==========================================================
# Reporting
# ==========================================================

class TestReporting:
    def _records(self):
        return [
            make_record(scene_id="s1", label="person", frame_id=0,
                        predicted_distance_m=2.1,
                        ground_truth_distance_m=2.0),
            make_record(scene_id="s1", label="chair", frame_id=1,
                        predicted_distance_m=2.6,
                        ground_truth_distance_m=2.5),
            make_record(scene_id="s2", label="person", frame_id=2,
                        predicted_distance_m=None,
                        distance_provenance="UNAVAILABLE"),
        ]

    def test_analyze_dataset_shape(self):
        a = analyze_dataset(self._records())
        for key in ("dataset_summary", "validation_triage",
                    "provenance_summary", "overall_metrics",
                    "per_label_metrics", "per_scene_metrics",
                    "distance_bins", "error_distribution"):
            assert key in a
        assert a["navigation_summary"] is None
        # MEASURED absent -> explicitly reported as absent
        assert a["provenance_summary"]["MEASURED"]["count"] == 0
        assert a["provenance_summary"]["MEASURED"]["metrics"] is None
        assert a["provenance_summary"]["SIMULATED"]["validation_class"] == \
            "SIMULATION-VALIDATED"

    def test_analyze_with_events(self):
        events = [_event("STOP", [_person_obj(risk="CRITICAL")], frame=0)]
        a = analyze_dataset(self._records(), events)
        assert a["navigation_summary"]["action_counts"]["STOP"] == 1
        assert a["risk_summary"]["risk_level_counts"]["CRITICAL"] == 1
        assert a["temporal_summary"]["frames"] == 1

    def test_build_report_roles(self):
        a = analyze_dataset(self._records())
        r = build_report(a, dataset_role="VALIDATION", name="rep",
                         source="x.jsonl")
        assert r["dataset_role"] == "VALIDATION"
        assert r["report_schema_version"] == 1
        assert r["deterministic"] is True
        assert any("does not validate safety" in lim
                   for lim in r["limitations"])
        with pytest.raises(ValueError):
            build_report(a, dataset_role="TRAINING")

    def test_report_determinism(self):
        a = analyze_dataset(self._records())
        r1 = build_report(a, dataset_role="VALIDATION")
        r2 = build_report(analyze_dataset(self._records()),
                          dataset_role="VALIDATION")
        assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)

    def test_evaluate_pair_enforces_disjointness(self):
        calib = [make_record(scene_id="c", frame_id=i) for i in range(3)]
        val = [make_record(scene_id="v", frame_id=i) for i in range(3)]
        pair = evaluate_pair(calib, val)
        assert pair["calibration"]["dataset_role"] == "CALIBRATION"
        assert pair["validation"]["dataset_role"] == "VALIDATION"
        assert "VERIFIED" in pair["disjointness"]

    def test_evaluate_pair_fails_loudly_on_overlap(self):
        overlap = [make_record(scene_id="s", frame_id=1, track_id=1)]
        calib = overlap
        val = [make_record(scene_id="s", frame_id=1, track_id=1)]
        with pytest.raises(ValueError, match="overlap"):
            evaluate_pair(calib, val)

    def test_markdown_rendering_contains_sections(self):
        report = build_report(
            analyze_dataset(self._records(),
                            [_event("STOP", [_person_obj()], frame=0)]),
            dataset_role="VALIDATION")
        md = render_markdown(report)
        for heading in ("# sina_evaluation_report",
                        "## Dataset summary", "## Provenance summary",
                        "## Overall metrics", "## Distance bins",
                        "## Error distribution", "## Per-label metrics",
                        "## Navigation summary", "## Risk summary"):
            assert heading in md
        assert "ANALYSIS BINS" in md
        assert "does not validate safety" in md

    def test_markdown_metric_formatting(self):
        report = build_report(
            analyze_dataset([make_record(predicted_distance_m=1.5,
                                         ground_truth_distance_m=1.0)]))
        md = render_markdown(report)
        assert "0.500" in md   # _fmt float formatting

    def test_csv_summary_rows(self):
        report = build_report(analyze_dataset(self._records()))
        rows = csv_summary_rows(report)
        assert rows[0][:3] == ["section", "name", "sample_count"]
        sections = {r[0] for r in rows[1:]}
        assert sections == {"distance_bin", "label"}

    def test_writers(self, tmp_path):
        report = build_report(analyze_dataset(self._records()),
                              dataset_role="CALIBRATION")
        jp = tmp_path / "rep.json"
        mp = tmp_path / "rep.md"
        cp = tmp_path / "rep.csv"
        write_json_report(jp, report)
        write_markdown_report(mp, report)
        write_csv_summary(cp, report)
        loaded = json.loads(jp.read_text(encoding="utf-8"))
        assert loaded["dataset_role"] == "CALIBRATION"
        assert "## Overall metrics" in mp.read_text(encoding="utf-8")
        assert cp.read_text(encoding="utf-8").startswith("section,name")


# ==========================================================
# CLI
# ==========================================================

def _run_cli(args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "evaluation", *args],
        capture_output=True, text=True, cwd=cwd,
        env=None,
    )


class TestCLI:
    def test_analyze_success_and_outputs(self, tmp_path):
        recs = [make_record(scene_id="s1", frame_id=i) for i in range(5)]
        rp = _write_records(tmp_path / "records.jsonl", recs)
        result = _run_cli([
            "analyze", rp,
            "--role", "VALIDATION",
            "--json", str(tmp_path / "out.json"),
            "--markdown", str(tmp_path / "out.md"),
            "--csv", str(tmp_path / "out.csv"),
        ])
        assert result.returncode == 0, result.stderr
        assert "records" in result.stdout
        assert (tmp_path / "out.json").exists()
        assert (tmp_path / "out.md").exists()
        assert (tmp_path / "out.csv").exists()
        loaded = json.loads((tmp_path / "out.json").read_text("utf-8"))
        assert loaded["dataset_role"] == "VALIDATION"

    def test_malformed_input_error_exit(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text("{broken}\n", encoding="utf-8")
        result = _run_cli(["analyze", str(p)])
        assert result.returncode == 2
        assert "line 1" in result.stderr
        assert "Traceback" not in result.stderr

    def test_missing_file_error_exit(self, tmp_path):
        result = _run_cli(["analyze", str(tmp_path / "missing.jsonl")])
        assert result.returncode == 2
        assert "not found" in result.stderr

    def test_empty_dataset_error_exit(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("\n\n", encoding="utf-8")
        result = _run_cli(["analyze", str(p)])
        assert result.returncode == 2
        assert "no records" in result.stderr

    def test_events_option(self, tmp_path):
        recs = [make_record()]
        rp = _write_records(tmp_path / "records.jsonl", recs)
        ep = tmp_path / "events.jsonl"
        ep.write_text(json.dumps(
            _event("STOP", [_person_obj(risk="CRITICAL")])) + "\n",
            encoding="utf-8")
        result = _run_cli(["analyze", rp, "--events", str(ep)])
        assert result.returncode == 0
        assert "STOP" in result.stdout

    def test_no_traceback_on_bad_events(self, tmp_path):
        rp = _write_records(tmp_path / "records.jsonl", [make_record()])
        ep = tmp_path / "events.jsonl"
        ep.write_text("not json\n", encoding="utf-8")
        result = _run_cli(["analyze", rp, "--events", str(ep)])
        assert result.returncode == 2
        assert "Traceback" not in result.stderr
