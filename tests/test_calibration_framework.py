"""
tests/test_calibration_framework.py

PHASE 10 PREPARATION - CALIBRATION EXPERIMENT FRAMEWORK TESTS.
HARDWARE-FREE. All fixtures are SIMULATED (provenance "SIMULATED",
scene ids prefixed "sim_") - nothing here is or claims to be MEASURED
OAK-D data, and no real-world accuracy is claimed.

Covers (milestone §23):

  - experiment session metadata (round-trip, unknown fields, tags)
  - target-distance plans (validation, checklist, non-threshold labeling)
  - repeated observations
  - raw-data preservation (raw + calibrated side by side)
  - baseline metrics (always present, §10)
  - calibration model interface (fit/transform/frozen)
  - calibration fitting (affine closed form on synthetic bias)
  - transformation (deterministic mapping)
  - calibration/validation disjointness (overlap fails loudly)
  - frozen calibration model (no refit possible, §20)
  - validation isolation (inputs never mutated)
  - insufficient coverage (loud CalibrationError, §12)
  - per-label / per-scene analysis (§14/§12)
  - distance coverage (§15: targets vs truth kept separate)
  - invalid rate (§16, per dimension)
  - outlier reporting + auditable filtering (§17)
  - repeatability (§18, insufficient groups flagged)
  - deterministic reports (§21, byte-identical JSON)

Failure cases are tested explicitly. No OAK-D, no DepthAI, no network.
"""

import json
from pathlib import Path

import pytest

from evaluation.record import MeasurementRecord
from evaluation.metrics import assert_disjoint

from calibration import (
    AffineCalibration,
    CalibrationError,
    CalibrationExperiment,
    DEFAULT_EXPERIMENT_TARGETS_M,
    ExperimentSession,
    FilterResult,
    IdentityCalibration,
    TargetDistancePlan,
    apply_calibration,
    availability_breakdown,
    build_experiment_report,
    compare_candidates,
    coverage_report,
    evaluate_calibrated,
    filter_outliers,
    outlier_report,
    repeatability_analysis,
    render_experiment_markdown,
)


# ==============================================================
# Fixtures - ALL SIMULATED
# ==============================================================

def mkrec(
    scene: str = "sim_room",
    frame: int = 0,
    label: str = "person",
    pred=None,
    gt=None,
    prov: str = "SIMULATED",
    track=None,
    timestamp: float = 0.0,
    source: str = "mock_bbox",
) -> MeasurementRecord:
    return MeasurementRecord(
        schema_version=1,
        scene_id=scene,
        frame_id=frame,
        timestamp=timestamp,
        track_id=track,
        label=label,
        confidence=0.9,
        bbox=(100, 100, 200, 300),
        region="CENTER",
        predicted_distance_m=pred,
        distance_provenance=prov,
        distance_source=source,
        ground_truth_distance_m=gt,
    )


def biased_dataset(
    scale: float = 1.1,
    offset: float = 0.0,
    scenes=("sim_a", "sim_b"),
    targets=DEFAULT_EXPERIMENT_TARGETS_M,
    reps: int = 3,
    gt_jitter=(0.0, -0.01, 0.01),
):
    """
    SIMULATED dataset with a KNOWN systematic bias: predicted =
    scale * truth + offset (+ tiny deterministic jitter so identity
    and affine are distinguishable but well-conditioned).
    """
    records = []
    frame = 0
    for scene in scenes:
        for t in targets:
            for i in range(reps):
                gt = t + gt_jitter[i % len(gt_jitter)]
                pred = scale * gt + offset
                records.append(mkrec(
                    scene=scene, frame=frame, label="person",
                    pred=round(pred, 4), gt=round(gt, 4), track=frame,
                ))
                frame += 1
    return records


@pytest.fixture
def sim_split():
    """Disjoint SIMULATED calibration/validation sets with known bias."""
    calib = biased_dataset(scenes=("sim_a",))
    val = biased_dataset(scenes=("sim_b",))
    return calib, val


# ==============================================================
# Session metadata (§5)
# ==============================================================

class TestSession:
    def test_round_trip(self):
        s = ExperimentSession(
            session_id="sess_001", date_time="2026-09-14T10:00:00",
            device_id="184430102190ED0F00", device_model="OAK-D Lite (RVC2)",
            software_version="b30ef83", depthai_version="3.7.1",
            rgb_resolution="640x400", stereo_resolution="640x480", fps=15,
            scene_id="sim_a", lighting="indoor, artificial",
            surfaces="matte floor", target_distance_m=2.0,
            operator_notes="tape measure, eye-level", tags=("sim", "bench"),
        )
        parsed = ExperimentSession.parse_json(s.to_json())
        assert parsed == s

    def test_all_optional_defaults(self):
        s = ExperimentSession()
        assert s.session_id is None and s.tags == ()
        parsed = ExperimentSession.parse_json(s.to_json())
        assert parsed == s

    def test_unknown_field_rejected(self):
        with pytest.raises(ValueError, match="unknown session fields"):
            ExperimentSession.from_dict({"mystery": 1})

    def test_invalid_tags_rejected(self):
        with pytest.raises(ValueError, match="tags must be a list"):
            ExperimentSession.from_dict({"tags": "not-a-list"})

    def test_unknown_values_stay_none(self):
        s = ExperimentSession.from_dict({"session_id": "x"})
        assert s.device_id is None and s.target_distance_m is None


# ==============================================================
# Target-distance plan (§6)
# ==============================================================

class TestPlan:
    def test_defaults_are_experiment_targets(self):
        plan = TargetDistancePlan()
        assert plan.targets_m == DEFAULT_EXPERIMENT_TARGETS_M
        assert plan.repetitions >= 1

    def test_total_and_checklist(self):
        plan = TargetDistancePlan(targets_m=(1.0, 2.0), repetitions=5)
        assert plan.total_observations() == 10
        assert plan.checklist() == [(1.0, 5), (2.0, 5)]

    def test_summary_labels_not_thresholds(self):
        summary = TargetDistancePlan().summary()
        assert summary["kind"] == "EXPERIMENT TARGETS - not navigation/safety thresholds"
        assert summary["calibrated"] is False

    def test_invalid_targets_rejected(self):
        with pytest.raises(ValueError):
            TargetDistancePlan(targets_m=())
        with pytest.raises(ValueError):
            TargetDistancePlan(targets_m=(0.0, 1.0))
        with pytest.raises(ValueError):
            TargetDistancePlan(targets_m=(1.0, 1.0))
        with pytest.raises(ValueError):
            TargetDistancePlan(repetitions=0)


# ==============================================================
# Coverage (§13/§15) and availability (§16)
# ==============================================================

class TestCoverage:
    def test_distance_coverage_stats(self):
        recs = [mkrec(gt=1.0, pred=1.1), mkrec(gt=2.0, pred=2.2),
                mkrec(gt=3.0, pred=3.0)]
        cov = coverage_report(recs)
        dc = cov["distance_coverage"]
        assert dc["n_with_ground_truth"] == 3
        assert dc["min_m"] == 1.0 and dc["max_m"] == 3.0
        assert dc["unique_values"] == 3
        assert cov["scenes"] == ["sim_room"]
        assert cov["labels"] == ["person"]

    def test_requested_targets_kept_separate_from_truth(self):
        recs = [mkrec(gt=2.02, pred=2.0)]  # placement ≠ exact truth
        cov = coverage_report(recs, requested_targets_m=[2.0, 4.0])
        assert cov["requested_targets_m"] == [2.0, 4.0]
        # 2.0 was requested; observed truth 2.02 matches within tolerance.
        assert cov["targets_without_ground_truth"] == [4.0]
        # And the two are reported as DISTINCT keys (§15).
        assert "distance_coverage" in cov

    def test_empty_coverage(self):
        cov = coverage_report([])
        assert cov["distance_coverage"] is None
        assert cov["n_total"] == 0

    def test_availability_by_dimension(self):
        recs = [
            mkrec(gt=1.0, pred=1.0, label="person", scene="sim_a", prov="SIMULATED"),
            mkrec(gt=2.0, pred=None, prov="UNAVAILABLE", label="chair", scene="sim_a"),
            mkrec(gt=2.0, pred=2.0, label="chair", scene="sim_b", prov="SIMULATED"),
        ]
        av = availability_breakdown(recs)
        assert av["overall"]["ALL"]["n"] == 3
        assert av["overall"]["ALL"]["invalid_rate"] == pytest.approx(1 / 3)
        assert av["by_label"]["chair"]["invalid_rate"] == pytest.approx(1 / 2)
        assert av["by_scene"]["sim_a"]["invalid_rate"] == pytest.approx(1 / 2)
        assert av["by_provenance"]["UNAVAILABLE"]["invalid_rate"] == 1.0
        assert av["by_distance"]["2.00 m"]["n"] == 2

    def test_availability_empty(self):
        av = availability_breakdown([])
        assert av["overall"]["ALL"]["invalid_rate"] == 0.0


# ==============================================================
# Repeatability (§18)
# ==============================================================

class TestRepeatability:
    def test_sufficient_group_statistics(self):
        recs = [mkrec(gt=2.0, pred=p) for p in (2.0, 2.02, 1.98)]
        groups = repeatability_analysis(recs, min_samples=3)
        g = next(g for g in groups if g.group_key == "2.00 m")
        assert not g.insufficient
        assert g.n_valid == 3
        assert g.mean_m == pytest.approx(2.0, abs=1e-9)
        assert g.median_m == pytest.approx(2.0, abs=1e-9)
        assert g.min_m == 1.98 and g.max_m == 2.02
        assert g.std_m > 0

    def test_insufficient_group_flagged_no_stats(self):
        recs = [mkrec(gt=2.0, pred=2.0)]  # 1 valid < min_samples=3
        groups = repeatability_analysis(recs, min_samples=3)
        assert len(groups) == 1
        assert groups[0].insufficient is True
        assert groups[0].mean_m is None and groups[0].std_m is None

    def test_group_by_label_and_scene(self):
        recs = [
            mkrec(label="person", scene="sim_a", gt=1.0, pred=1.0),
            mkrec(label="person", scene="sim_a", gt=1.0, pred=1.0),
            mkrec(label="person", scene="sim_a", gt=1.0, pred=1.0),
            mkrec(label="chair", scene="sim_b", gt=2.0, pred=2.0),
        ]
        by_label = repeatability_analysis(recs, group_by="label", min_samples=3)
        person = next(g for g in by_label if g.group_key == "person")
        chair = next(g for g in by_label if g.group_key == "chair")
        assert not person.insufficient and chair.insufficient

        by_scene = repeatability_analysis(recs, group_by="scene_id", min_samples=3)
        sim_a = next(g for g in by_scene if g.group_key == "sim_a")
        assert not sim_a.insufficient

    def test_invalid_predictions_excluded(self):
        recs = [
            mkrec(gt=2.0, pred=2.0), mkrec(gt=2.0, pred=2.0),
            mkrec(gt=2.0, pred=None, prov="UNAVAILABLE"),
        ]
        groups = repeatability_analysis(recs, min_samples=3)
        assert groups[0].insufficient  # only 2 valid


# ==============================================================
# Outliers (§17) and auditable filtering
# ==============================================================

class TestOutliers:
    def test_largest_errors_listed_descriptively(self):
        recs = [mkrec(gt=2.0, pred=2.0), mkrec(gt=2.0, pred=3.0),
                mkrec(gt=5.0, pred=5.05)]
        rep = outlier_report(recs)
        assert rep["n_valid"] == 3
        largest = rep["largest_absolute_errors"]
        assert largest[0]["abs_error_m"] == pytest.approx(1.0)
        assert largest[0]["scene_id"] == "sim_room"
        # Listing only: report carries no removal count/filter result.
        assert "removed" not in json.dumps(rep)

    def test_no_records_no_crash(self):
        rep = outlier_report([])
        assert rep["n_valid"] == 0 and rep["largest_absolute_errors"] == []

    def test_error_by_distance_bin(self):
        recs = [mkrec(gt=0.5, pred=0.6), mkrec(gt=4.5, pred=4.6)]
        rep = outlier_report(recs)
        bins = rep["abs_error_by_distance_bin"]
        assert bins["0-1 m"]["mean_abs_error_m"] == pytest.approx(0.1)
        assert bins["3-5 m"]["mean_abs_error_m"] == pytest.approx(0.1)

    def test_filter_outliers_is_auditable(self):
        recs = [mkrec(gt=2.0, pred=2.0), mkrec(gt=2.0, pred=2.02),
                mkrec(gt=2.0, pred=5.0)]
        before = outlier_report(recs)["abs_error_distribution"]
        res = filter_outliers(recs, max_abs_error_m=0.5)
        assert isinstance(res, FilterResult)
        assert res.rule == "absolute error <= 0.5 m"
        assert res.n_before == 3 and res.n_removed == 1 and res.n_after == 2
        assert res.mae_before_m == pytest.approx(sum(
            abs(r.predicted_distance_m - r.ground_truth_distance_m)
            for r in recs) / 3)
        assert res.mae_after_m < res.mae_before_m
        assert all(
            abs(r.predicted_distance_m - r.ground_truth_distance_m) <= 0.5
            for r in res.kept
        )

    def test_filter_never_mutates_raw(self):
        recs = [mkrec(gt=2.0, pred=5.0)]
        filter_outliers(recs, max_abs_error_m=0.5)
        assert recs[0].predicted_distance_m == 5.0  # untouched

    def test_filter_invalid_rule_rejected(self):
        with pytest.raises(ValueError):
            filter_outliers([], max_abs_error_m=0.0)


# ==============================================================
# Calibration models (§9-§12)
# ==============================================================

class TestModels:
    def test_identity_transform(self):
        fm = IdentityCalibration().fit(biased_dataset())
        assert fm.transform(2.5) == 2.5
        assert fm.is_identity

    def test_affine_recovers_known_bias(self):
        recs = biased_dataset(scale=1.1, offset=0.05)
        fm = AffineCalibration().fit(recs)
        # truth = a*pred + b should invert y = 1.1x + 0.05
        a_inv, b_inv = 1 / 1.1, -0.05 / 1.1
        for probe in (1.0, 2.5, 5.0):
            expected = a_inv * probe + b_inv
            assert fm.transform(probe) == pytest.approx(expected, abs=1e-6)

    def test_affine_beats_identity_on_biased_data(self):
        recs = biased_dataset(scale=1.15)
        comparison, _ = compare_candidates(
            [IdentityCalibration(), AffineCalibration()], recs
        )
        by_name = {e.name: e for e in comparison.evaluations}
        assert by_name["affine"].mae_m < by_name["identity"].mae_m
        assert comparison.selected == "affine"

    def test_identity_wins_on_unbiased_data(self):
        # pred == gt exactly -> affine's least-squares fit is y=x with
        # float noise; identity must win the documented tie-break.
        recs = [mkrec(gt=t, pred=t) for t in (1.0, 2.0, 3.0)]
        comparison, frozen = compare_candidates(
            [IdentityCalibration(), AffineCalibration()], recs
        )
        assert frozen.is_identity
        assert comparison.selected == "identity"

    def test_insufficient_samples_fail_loudly(self):
        with pytest.raises(CalibrationError, match="insufficient calibration"):
            AffineCalibration().fit([mkrec(gt=2.0, pred=2.0)])
        with pytest.raises(CalibrationError, match="insufficient calibration"):
            AffineCalibration().fit([])

    def test_unusable_records_do_not_count_as_samples(self):
        recs = [mkrec(gt=2.0, pred=2.0),
                mkrec(gt=2.0, pred=None, prov="UNAVAILABLE")]
        with pytest.raises(CalibrationError):
            AffineCalibration().fit(recs)

    def test_frozen_model_has_no_refit(self):
        fm = AffineCalibration().fit(biased_dataset())
        assert not hasattr(fm, "fit")
        with pytest.raises(AttributeError):
            fm.fit([])  # type: ignore[attr-defined]

    def test_transform_before_fit_rejected(self):
        with pytest.raises(CalibrationError, match="no fitted parameters"):
            AffineCalibration().transform(2.0)

    def test_unfitted_identity_usable(self):
        # Identity has no parameters to estimate.
        assert IdentityCalibration().transform(3.3) == 3.3

    def test_selection_rule_documented(self):
        comparison, _ = compare_candidates([IdentityCalibration()],
                                           biased_dataset())
        assert "CALIBRATION" in comparison.selection_rule
        assert "never" in comparison.selection_rule


# ==============================================================
# Provenance integrity (§7/§22)
# ==============================================================

class TestProvenance:
    def test_identity_preserves_provenance_verbatim(self):
        fm = IdentityCalibration().fit([mkrec(gt=2.0, pred=2.0)])
        assert fm.calibrated_provenance_for("MEASURED") == "MEASURED"
        assert fm.calibrated_provenance_for("SIMULATED") == "SIMULATED"
        assert fm.calibrated_provenance_for("UNAVAILABLE") == "UNAVAILABLE"
        assert fm.calibrated_provenance_for("STALE") == "STALE"

    def test_fitted_transform_never_claims_measured(self):
        fm = AffineCalibration().fit(biased_dataset())
        assert fm.calibrated_provenance_for("MEASURED") == "MEASURED_CALIBRATED"
        assert fm.calibrated_provenance_for("MEASURED") != "MEASURED"
        assert fm.calibrated_provenance_for("SIMULATED") == "SIMULATED_CALIBRATED"
        # Status provenances pass through unchanged.
        assert fm.calibrated_provenance_for("UNAVAILABLE") == "UNAVAILABLE"
        assert fm.calibrated_provenance_for("STALE") == "STALE"

    def test_calibrated_observation_keeps_raw_and_calibrated(self, sim_split):
        calib, _ = sim_split
        fm = AffineCalibration().fit(calib)
        obs = apply_calibration(calib, fm)
        first = obs[0]
        raw = calib[0]
        assert first.raw_predicted_m == raw.predicted_distance_m
        assert first.raw_provenance == "SIMULATED"
        assert first.calibrated_predicted_m != raw.predicted_distance_m
        assert first.calibrated_provenance == "SIMULATED_CALIBRATED"
        assert first.ground_truth_m == raw.ground_truth_distance_m

    def test_none_prediction_stays_none(self):
        recs = [mkrec(gt=2.0, pred=2.0),
                mkrec(gt=2.0, pred=None, prov="UNAVAILABLE")]
        fm = IdentityCalibration().fit(recs)
        obs = apply_calibration(recs, fm)
        assert obs[0].calibrated_predicted_m == 2.0
        assert obs[1].calibrated_predicted_m is None

    def test_raw_records_never_mutated_by_experiment(self, sim_split):
        calib, val = sim_split
        snapshot_calib = [r.to_json() for r in calib]
        snapshot_val = [r.to_json() for r in val]
        CalibrationExperiment().run(calib, val)
        assert [r.to_json() for r in calib] == snapshot_calib
        assert [r.to_json() for r in val] == snapshot_val


# ==============================================================
# Experiment workflow (§3, §10, §19-§20)
# ==============================================================

class TestExperiment:
    def test_baseline_present_alongside_calibrated(self, sim_split):
        calib, val = sim_split
        result = CalibrationExperiment(
            candidates=[IdentityCalibration(), AffineCalibration()]
        ).run(calib, val)
        # §10: baseline ALWAYS reported, both datasets.
        assert result.baseline_validation.mae_m is not None
        assert result.baseline_calibration.mae_m is not None
        # Frozen-model validation improves on raw baseline for biased data.
        assert result.validation_metrics.mae_m < result.baseline_validation.mae_m

    def test_identity_only_run_equals_baseline(self, sim_split):
        calib, val = sim_split
        result = CalibrationExperiment(
            candidates=[IdentityCalibration()]
        ).run(calib, val)
        assert result.selected_model == "identity"
        assert result.validation_metrics.mae_m == pytest.approx(
            result.baseline_validation.mae_m
        )
        assert result.calibration_metrics.mae_m == pytest.approx(
            result.baseline_calibration.mae_m
        )

    def test_disjointness_enforced_before_fitting(self):
        calib = biased_dataset(scenes=("sim_a",))
        # Same scene AND same (frame, track) identity in both sets.
        val = [calib[0]] + biased_dataset(scenes=("sim_b",))[:5]
        with pytest.raises(ValueError, match="overlap detected"):
            CalibrationExperiment().run(calib, val)

    def test_disjoint_scenes_pass(self, sim_split):
        calib, val = sim_split
        assert_disjoint(calib, val)  # must not raise
        result = CalibrationExperiment().run(calib, val)
        assert result.n_calibration == len(calib)
        assert result.n_validation == len(val)

    def test_validation_metrics_from_frozen_model_only(self, sim_split):
        calib, val = sim_split
        exp = CalibrationExperiment(candidates=[AffineCalibration()])
        result = exp.run(calib, val)
        # If validation had leaked into fitting, a second run reusing
        # the validation set as calibration would change the model.
        r2 = exp.run(val, calib)
        assert result.selected_model == r2.selected_model == "affine"
        # Different data, same recovered inverse bias (deterministic fit).
        a1 = result.candidate_comparison["evaluations"]
        a2 = r2.candidate_comparison["evaluations"]
        assert a1 == a2

    def test_result_dict_serializable_and_deterministic(self, sim_split):
        calib, val = sim_split
        exp = CalibrationExperiment(
            candidates=[IdentityCalibration(), AffineCalibration()]
        )
        d1 = exp.run(calib, val).to_dict()
        d2 = exp.run(calib, val).to_dict()
        assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)

    def test_provenance_class_in_result(self, sim_split):
        calib, val = sim_split
        result = CalibrationExperiment().run(calib, val)
        assert result.validation_class == "SIMULATION-VALIDATED"


# ==============================================================
# Report (§21)
# ==============================================================

class TestReport:
    def _report(self, sim_split):
        calib, val = sim_split
        result = CalibrationExperiment(
            candidates=[IdentityCalibration(), AffineCalibration()]
        ).run(calib, val)
        plan = TargetDistancePlan()
        return build_experiment_report(
            result,
            calibration_records=calib,
            validation_records=val,
            requested_targets_m=list(plan.targets_m),
        )

    def test_report_sections_present(self, sim_split):
        report = self._report(sim_split)
        for section in (
            "metadata", "sample_counts", "provenance", "raw_baseline",
            "calibration_model", "calibration_performance",
            "validation_performance", "distance_coverage",
            "scene_coverage", "label_coverage", "invalid_rate",
            "error_distribution", "repeatability", "outliers",
            "limitations", "validation_classification",
        ):
            assert section in report, f"missing section {section}"

    def test_report_deterministic(self, sim_split):
        r1 = json.dumps(self._report(sim_split), sort_keys=True)
        r2 = json.dumps(self._report(sim_split), sort_keys=True)
        assert r1 == r2

    def test_calibration_vs_validation_performance_separate(self, sim_split):
        report = self._report(sim_split)
        assert "calibration_performance" in report
        assert "validation_performance" in report
        assert report["raw_baseline"]["note"].startswith("UNCALIBRATED BASELINE")

    def test_validation_classification_honest(self, sim_split):
        report = self._report(sim_split)
        vc = report["validation_classification"]
        assert vc["real_oak_d_hardware"].startswith("NOT VALIDATED")
        assert vc["real_distance_accuracy"] == "NOT VALIDATED"
        assert vc["safety"] == "NOT VALIDATED"
        assert "BLOCKED" in report["metadata"]["hardware_validation"]

    def test_markdown_renders_key_sections(self, sim_split):
        md = render_experiment_markdown(self._report(sim_split))
        assert "Phase 10 preparation" in md
        assert "UNCALIBRATED BASELINE" in md
        assert "NOT VALIDATED" in md
        assert "BLOCKED" in md

    def test_minimal_report_without_records(self, sim_split):
        calib, val = sim_split
        result = CalibrationExperiment().run(calib, val)
        report = build_experiment_report(result)
        assert "distance_coverage" not in report  # enrichment absent
        assert report["metadata"]["selected_model"] == "identity"

    def test_outlier_filter_in_report(self, sim_split):
        calib, val = sim_split
        result = CalibrationExperiment().run(calib, val)
        fr = filter_outliers(calib + val, max_abs_error_m=10.0)
        report = build_experiment_report(
            result, calibration_records=calib, validation_records=val,
            outlier_filter=fr,
        )
        assert report["outlier_filtering"]["n_removed"] == 0

    def test_report_writes_to_disk(self, sim_split, tmp_path: Path):
        report = self._report(sim_split)
        p = tmp_path / "report.json"
        p.write_text(json.dumps(report, sort_keys=True, indent=2))
        loaded = json.loads(p.read_text())
        assert loaded["metadata"] == report["metadata"]


# ==============================================================
# Source-level guards (§3/§6: no navigation thresholds here)
# ==============================================================

class TestSourceGuards:
    def test_calibration_package_imports_no_navigation_config(self):
        """AST-level guard: calibration modules must not import anything
        from navigation or risk policy (prose mentions are irrelevant)."""
        import ast
        import calibration
        from pathlib import Path
        pkg_dir = Path(calibration.__file__).parent
        offenders = []
        for py in pkg_dir.glob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.split(".")[0] == "navigation":
                        offenders.append(py.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] == "navigation":
                            offenders.append(py.name)
        assert offenders == [], (
            f"calibration modules must not import navigation config: {offenders}"
        )
