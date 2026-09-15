"""
tests/test_measurement_contract.py

PHASE 12 - PHYSICAL MEASUREMENT CONTRACT & CALIBRATION READINESS.
HARDWARE-FREE. All fixtures are SIMULATED or measured-CONTRACT
synthetic doubles; nothing here validates OAK-D hardware, physical
distance accuracy, calibration generalization, or safety.

Covers (milestone §16 + §4/§5/§6/§7/§8/§9/§10/§11/§12/§13/§14/§15):

  1  valid measured observation            13 unavailable exclusion
  2  simulated provenance preservation     14 stale exclusion
  3  unavailable preservation              15 missing truth exclusion
  4  stale rejection (boundary)            16 planned target vs truth
  5  invalid distance rejection            17 split leakage rejection
  6  zero distance rejection               18 provenance survives JSON
  7  negative distance rejection           19 provenance survives manifest
  8  invalid timestamp handling            20 measurement quality summary
  9  calibration provenance derivation     21 software readiness PASS
  10 raw record immutability               22 hardware status separate
  11 measured calibration output           23 SIMULATED never relabeled
  12 simulated calibration output          24 calibrated never overwrites raw
  +  invariants A-G, provider contract, fusion contract, eligibility,
     dataset acceptance, full-pipeline provenance flow
"""

import json

import pytest

from depth.provider import DepthMeasurement, DistanceProvenance
from evaluation.injection import ScriptedDepthProvider
from evaluation.metrics import assert_disjoint, evaluate
from evaluation.record import MeasurementRecord
from evaluation.reporting import measurement_quality_summary
from vision.detection_manager import DetectionManager
from vision.distance_fusion import DistanceFusion
from vision.object_detector import BoundingBox, DetectedObject

from calibration.contract import (
    ALL_PROVENANCES,
    CALIBRATED_PROVENANCES,
    SOURCE_PROVENANCES,
    VERDICT_INVALID,
    VERDICT_STALE,
    VERDICT_UNAVAILABLE,
    VERDICT_VALID,
    accept_dataset,
    calibration_eligibility,
    check_observation,
    provenance_invariant_report,
    software_calibration_readiness,
)
from calibration.capture import COMPLETED
from calibration.experiment import (
    CalibrationExperiment,
    apply_calibration,
    evaluate_calibrated,
)
from calibration.models import (
    AffineCalibration,
    IdentityCalibration,
)
from calibration.plan import TargetDistancePlan
from calibration.protocol import (
    CALIBRATION,
    VALIDATION,
    ExperimentProtocol,
    ProtocolFrozenError,
)


# --------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------

def rec(
    scene="sim_contract", frame=1, track=1, pred=2.0, gt=2.0,
    provenance="SIMULATED", ts=100.0, source="contract-test",
) -> MeasurementRecord:
    return MeasurementRecord(
        schema_version=1, scene_id=scene, frame_id=frame,
        timestamp=ts, track_id=track, label="person",
        confidence=0.9, bbox=(0, 0, 40, 120), region="CENTER",
        predicted_distance_m=pred, distance_provenance=provenance,
        distance_source=source, ground_truth_distance_m=gt,
    )


def det(label="person", conf=0.95, bbox=(0, 0, 40, 120)) -> DetectedObject:
    x1, y1, x2, y2 = bbox
    return DetectedObject(
        label=label, confidence=conf, bbox=BoundingBox(x1, y1, x2, y2),
    )


# --------------------------------------------------------------
# 1-3. Observation-level contract verdicts (§3)
# --------------------------------------------------------------

class TestObservationContract:
    def test_valid_measured_observation(self):
        v = check_observation(rec(provenance="MEASURED"), now=100.5)
        assert v.classification == VERDICT_VALID
        assert v.ok and v.reasons == ()

    def test_valid_simulated_observation(self):
        v = check_observation(rec(provenance="SIMULATED"), now=100.5)
        assert v.classification == VERDICT_VALID

    def test_unavailable_preserved(self):
        r = rec(provenance="UNAVAILABLE", pred=None, gt=None)
        v = check_observation(r, now=100.5)
        assert v.classification == VERDICT_UNAVAILABLE

    def test_stale_provenance_preserved(self):
        r = rec(provenance="STALE", pred=None, gt=None)
        v = check_observation(r, now=100.5)
        assert v.classification == VERDICT_STALE

    def test_stale_label_cannot_carry_a_value(self):
        r = rec(provenance="STALE", pred=2.0)
        v = check_observation(r, now=100.5)
        assert v.classification == VERDICT_INVALID
        assert "cannot carry a value" in v.reasons[0]

    def test_unavailable_label_cannot_carry_a_value(self):
        r = rec(provenance="UNAVAILABLE", pred=2.0)
        v = check_observation(r, now=100.5)
        assert v.classification == VERDICT_INVALID

    def test_verdict_never_mutates_the_record(self):
        r = rec(provenance="STALE", pred=2.0)
        snapshot = r.to_json()
        check_observation(r, now=100.5)
        assert r.to_json() == snapshot


# --------------------------------------------------------------
# 4-8. Stale boundary + invalid numerics (§5)
# --------------------------------------------------------------

class TestStaleAndInvalid:
    STALE_S = 2.0

    def test_exactly_within_stale_threshold_is_valid(self):
        # age == stale_after_s is NOT beyond the window (fusion uses
        # strict >): boundary is VALID.
        r = rec(provenance="MEASURED", ts=100.0)
        v = check_observation(r, now=100.0 + self.STALE_S)
        assert v.classification == VERDICT_VALID

    def test_exactly_beyond_stale_threshold_is_stale(self):
        r = rec(provenance="MEASURED", ts=100.0)
        v = check_observation(r, now=100.0 + self.STALE_S + 0.001)
        assert v.classification == VERDICT_STALE
        assert v.reasons[0].startswith("stale:")

    def test_future_timestamp_rejected(self):
        r = rec(provenance="MEASURED", ts=101.0)
        v = check_observation(r, now=100.0)
        assert v.classification == VERDICT_INVALID
        assert "future timestamp" in v.reasons[0]

    def test_missing_timestamp_rejected(self):
        # A "missing" timestamp in this schema materializes as 0.0;
        # with now supplied, age = now - 0 exceeds the window -> STALE
        # (too old, not invalid). With now=None the 0.0 is rejected by
        # the capture-layer completeness rule instead (negative/absent
        # wall clocks are never fabricated here).
        r = rec(provenance="MEASURED", ts=0.0)
        v = check_observation(r, now=100.0)
        assert v.classification == VERDICT_STALE
        assert v.reasons[0].startswith("stale:")

    def test_non_finite_timestamp_rejected(self):
        r = rec(provenance="MEASURED", ts=float("nan"))
        v = check_observation(r, now=100.0)
        assert v.classification == VERDICT_INVALID
        assert "non-finite timestamp" in v.reasons[0]

    def test_non_finite_distance_rejected(self):
        r = rec(provenance="MEASURED", pred=float("inf"))
        v = check_observation(r, now=100.0)
        assert v.classification == VERDICT_INVALID
        assert "non-finite predicted distance" in v.reasons[0]

    def test_zero_distance_rejected_not_clamped(self):
        v = check_observation(rec(provenance="MEASURED", pred=0.0), now=100.0)
        assert v.classification == VERDICT_INVALID
        assert "non-positive" in v.reasons[0]

    def test_negative_distance_rejected_not_clamped(self):
        v = check_observation(rec(provenance="MEASURED", pred=-1.5), now=100.0)
        assert v.classification == VERDICT_INVALID
        assert "non-positive" in v.reasons[0]

    def test_unknown_provenance_rejected(self):
        v = check_observation(rec(provenance="GUESSED"), now=100.0)
        assert v.classification == VERDICT_INVALID
        assert "unknown provenance" in v.reasons[0]


# --------------------------------------------------------------
# Provider contract through the public DepthProvider API (§6)
# --------------------------------------------------------------

class TestProviderContract:
    def _provider(self, spec) -> ScriptedDepthProvider:
        return ScriptedDepthProvider(
            [{"person": spec}], name="contract-fixture", stale_age_s=2.0
        )

    def _measurement(self, provider) -> DepthMeasurement:
        provider.update()
        return provider.get_measurement((0, 0, 40, 120), (400, 640), "person",
                                        now=100.0)

    def test_synthetic_spec_is_simulated(self):
        m = self._measurement(self._provider(2.0))
        assert m.provenance_str == "SIMULATED" and m.usable

    def test_unavailable_spec(self):
        m = self._measurement(self._provider(None))
        assert m.provenance_str == "UNAVAILABLE"
        assert m.distance_m is None and not m.usable

    def test_stale_spec_carries_no_value(self):
        m = self._measurement(self._provider(("STALE", 2.0)))
        assert m.provenance_str == "STALE"
        assert m.distance_m is None  # stale value never masquerades

    def test_measured_contract_fixture(self):
        m = self._measurement(self._provider(("MEASURED", 2.0)))
        assert m.provenance_str == "MEASURED" and m.usable

    def test_old_spec_demoted_by_fusion_defense(self):
        # ("OLD", v): SIMULATED with an old timestamp (age 5 s by the
        # provider's default) - fusion must demote it to STALE (defense
        # in depth, restated as the contract). The default 2 s stale
        # window in TestProviderContract._provider is intentionally NOT
        # used here: age == window is not beyond it (strict >).
        provider = ScriptedDepthProvider(
            [{"person": ("OLD", 2.0)}], name="contract-fixture"
        )
        fusion = DistanceFusion(provider, clock=lambda: 100.0)
        dets = [det()]
        fusion.update()
        fused = fusion.fuse(dets, (400, 640))[0]
        assert fused.distance_provenance == "STALE"
        assert fused.distance is None  # stale value withheld


# --------------------------------------------------------------
# DistanceFusion contract (§7)
# --------------------------------------------------------------

class TestFusionContract:
    def _fuse(self, spec, prefill=None):
        provider = ScriptedDepthProvider([{"person": spec}],
                                         name="contract-fixture")
        fusion = DistanceFusion(provider, clock=lambda: 100.0,
                                allow_simulated=True)
        d = det()
        if prefill is not None:
            d.distance = prefill  # legacy detectors may pre-fill
        fusion.update()
        fused = fusion.fuse([d], (400, 640))[0]
        return fused, fusion

    def test_simulated_distance_and_provenance_preserved(self):
        fused, _ = self._fuse(2.0)
        assert fused.distance == pytest.approx(2.0)
        assert fused.distance_provenance == "SIMULATED"
        assert fused.distance_source == "contract-fixture"

    def test_measured_distance_and_provenance_preserved(self):
        fused, _ = self._fuse(("MEASURED", 1.7))
        assert fused.distance == pytest.approx(1.7)
        assert fused.distance_provenance == "MEASURED"
        assert fused.distance_source == "contract-fixture"

    def test_unavailable_leaves_no_usable_distance(self):
        fused, _ = self._fuse(None)
        assert fused.distance_provenance == "UNAVAILABLE"

    def test_unavailable_does_not_zero_a_previous_value(self):
        fused, _ = self._fuse(None, prefill=2.3)
        assert fused.distance == pytest.approx(2.3)  # kept, honestly stamped
        assert fused.distance_provenance == "UNAVAILABLE"

    def test_stale_value_never_reused_as_current(self):
        fused, summary = self._fuse(("STALE", 2.0))
        assert fused.distance_provenance == "STALE"
        assert fused.distance is None  # pre-filled value withheld
        assert summary.health()["stale_total"] == 1

    def test_old_timestamp_demoted_to_stale(self):
        fused, summary = self._fuse(("OLD", 2.0))
        assert fused.distance_provenance == "STALE"
        assert fused.distance is None
        assert summary.health()["stale_total"] == 1


# --------------------------------------------------------------
# 9. Calibration provenance derivation (§4 D/E/G)
# --------------------------------------------------------------

class TestCalibrationProvenance:
    def _fit_affine(self):
        rows = [
            rec(frame=1, provenance="MEASURED", pred=1.1, gt=1.0),
            rec(frame=2, provenance="MEASURED", pred=2.2, gt=2.0),
            rec(frame=3, provenance="MEASURED", pred=3.3, gt=3.0),
        ]
        return AffineCalibration().fit(rows)

    def test_measured_calibrated_output(self):
        frozen = self._fit_affine()
        obs = apply_calibration([rec(provenance="MEASURED", pred=2.2)],
                                frozen)[0]
        assert obs.calibrated_provenance == "MEASURED_CALIBRATED"

    def test_simulated_calibrated_output(self):
        frozen = self._fit_affine()
        obs = apply_calibration([rec(provenance="SIMULATED", pred=2.2)],
                                frozen)[0]
        assert obs.calibrated_provenance == "SIMULATED_CALIBRATED"

    def test_unavailable_and_stale_pass_through_unchanged(self):
        frozen = self._fit_affine()
        obs = apply_calibration(
            [rec(provenance="UNAVAILABLE", pred=None),
             rec(provenance="STALE", pred=None)],
            frozen,
        )
        assert [o.calibrated_provenance for o in obs] == \
            ["UNAVAILABLE", "STALE"]
        assert [o.calibrated_predicted_m for o in obs] == [None, None]

    def test_identity_keeps_vocabulary_verbatim(self):
        frozen = IdentityCalibration().fit([rec(provenance="MEASURED")])
        obs = apply_calibration([rec(provenance="SIMULATED", pred=2.0)],
                                frozen)[0]
        assert obs.calibrated_provenance == "SIMULATED"  # identity: no relabel

    def test_invariant_G_no_manual_relabeling_path(self):
        frozen = self._fit_affine()
        obs = apply_calibration(
            [rec(provenance="MEASURED", pred=2.2),
             rec(provenance="SIMULATED", pred=2.2),
             rec(provenance="UNAVAILABLE", pred=None),
             rec(provenance="STALE", pred=None)],
            frozen,
        )
        report = provenance_invariant_report([], calibrated=obs)
        assert report["invariants"]["G_provenance_derived_not_relabeled"]
        assert report["violations"] == []


# --------------------------------------------------------------
# 10/24. Raw record immutability (§10, §4 F)
# --------------------------------------------------------------

class TestRawImmutability:
    def test_calibrated_output_cannot_overwrite_raw(self):
        r = rec(provenance="MEASURED", pred=2.2, gt=2.0, frame=9)
        snapshot = r.to_json()
        frozen = AffineCalibration().fit(
            [rec(frame=1, provenance="MEASURED", pred=1.1, gt=1.0),
             rec(frame=2, provenance="MEASURED", pred=2.2, gt=2.0),
             rec(frame=3, provenance="MEASURED", pred=3.3, gt=3.0)]
        )
        apply_calibration([r], frozen)
        assert r.to_json() == snapshot  # raw untouched by calibration

    def test_deep_snapshot_across_full_stack(self):
        """Snapshot -> fit/diagnostics/report/manifest/rehydrate -> same."""
        import copy

        plan = TargetDistancePlan(targets_m=(1.0, 2.0), repetitions=1)
        protocol = ExperimentProtocol(
            plan=plan, scenes=("sim_contract",), experiment_id="p12"
        )
        captures = []
        for target, role in ((1.0, CALIBRATION), (2.0, VALIDATION)):
            c = protocol.start_capture("sim_contract", target, 1)
            rows = [
                rec(frame=int(target * 10) + i, track=1,
                    pred=target * 1.1 + i * 0.1, gt=target,
                    provenance="SIMULATED")
                for i in range(2)  # affine needs >= 2 calibration rows
            ]
            c.add_records(rows)
            c.attach_ground_truth({
                ("sim_contract", int(target * 10) + i, 1): target
                for i in range(2)
            })
            c.complete()
            protocol.assign(c, role)
            captures.append(c)

        raw_snapshot = [
            copy.deepcopy(r.to_dict())
            for c in captures for r in c.records
        ]

        cal = protocol.dataset_records(CALIBRATION)
        val = protocol.dataset_records(VALIDATION)
        # calibration + diagnostics + report + freeze + manifest
        experiment = CalibrationExperiment(
            candidates=[IdentityCalibration(), AffineCalibration()]
        ).run(cal, val)
        frozen = AffineCalibration().fit(cal)
        apply_calibration(cal, frozen)   # derived views built; raw must not move
        apply_calibration(val, frozen)
        from calibration.diagnostics import outlier_report
        outlier_report(cal + val)
        protocol.freeze()
        rehydrated = ExperimentProtocol.from_manifest(
            json.loads(json.dumps(protocol.manifest()))
        )
        assert rehydrated.frozen

        now_raw = [r.to_dict() for c in captures for r in c.records]
        assert now_raw == raw_snapshot  # semantic equality (§10)

    def test_experiment_inputs_untouched_by_run(self):
        cal = [rec(frame=1, pred=1.1, gt=1.0),
               rec(frame=2, pred=2.2, gt=2.0)]
        val = [rec(frame=11, pred=1.1, gt=1.0),
               rec(frame=12, pred=2.2, gt=2.0)]
        cal_json = [r.to_json() for r in cal]
        val_json = [r.to_json() for r in val]
        CalibrationExperiment(
            candidates=[IdentityCalibration(), AffineCalibration()]
        ).run(cal, val)
        assert [r.to_json() for r in cal] == cal_json
        assert [r.to_json() for r in val] == val_json


# --------------------------------------------------------------
# Provenance invariants A/B/C (§4)
# --------------------------------------------------------------

class TestProvenanceInvariants:
    def test_invariants_hold_on_contract_dataset(self):
        rows = [
            rec(frame=1, provenance="MEASURED", pred=1.1, gt=1.0),
            rec(frame=2, provenance="SIMULATED", pred=1.1, gt=1.0),
            rec(frame=3, provenance="UNAVAILABLE", pred=None, gt=None),
            rec(frame=4, provenance="STALE", pred=None, gt=None),
        ]
        report = provenance_invariant_report(rows)
        assert all(report["invariants"].values()), report["violations"]

    def test_invariant_A_simulated_never_measured_end_to_end(self):
        """SIMULATED cannot become MEASURED through ANY evaluated path."""
        rows = [rec(frame=i, provenance="SIMULATED", pred=1.0 + i / 10,
                    gt=1.0) for i in range(1, 6)]
        val = [rec(frame=10 + i, provenance="SIMULATED", pred=1.0 + i / 10,
                   gt=1.0) for i in range(1, 5)]
        experiment = CalibrationExperiment(
            candidates=[IdentityCalibration(), AffineCalibration()]
        ).run(rows, val)
        # Raw provenance untouched...
        assert all(r.distance_provenance == "SIMULATED"
                   for r in rows + val)
        # ...derived labels are SIMULATED_CALIBRATED at best...
        obs = apply_calibration(val, AffineCalibration().fit(rows))
        assert all(o.calibrated_provenance in ("SIMULATED",
                                              "SIMULATED_CALIBRATED")
                   for o in obs)
        # ...and the whole-run invariant report is clean.
        report = provenance_invariant_report(
            rows + val,
            apply_calibration(rows + val,
                              AffineCalibration().fit(rows)),
        )
        assert report["invariants"]["A_simulated_never_measured"]

    def test_invariants_B_C_machine_report(self):
        rows = [rec(frame=1, provenance="UNAVAILABLE", pred=None),
                rec(frame=2, provenance="STALE", pred=None)]
        report = provenance_invariant_report(rows)
        assert report["invariants"]["B_unavailable_stays_unavailable"]
        assert report["invariants"]["C_stale_never_fresh"]
        assert report["violations"] == []


# --------------------------------------------------------------
# 13-15. Calibration eligibility (§8, §11)
# --------------------------------------------------------------

class TestCalibrationEligibility:
    def test_valid_pair_is_eligible(self):
        report = calibration_eligibility(
            [rec(provenance="MEASURED", pred=2.2, gt=2.0)], now=100.5
        )
        assert report.n_eligible == 1 and not report.excluded

    def test_unavailable_excluded(self):
        report = calibration_eligibility(
            [rec(provenance="UNAVAILABLE", pred=None, gt=2.0)], now=100.5
        )
        assert report.n_eligible == 0
        assert "unavailable" in report.excluded[0][3]

    def test_stale_excluded_from_fresh_fit(self):
        report = calibration_eligibility(
            [rec(provenance="MEASURED", pred=2.2, gt=2.0, ts=90.0)],
            now=100.0, stale_after_s=2.0,
        )
        assert report.n_eligible == 0
        assert "stale" in report.excluded[0][3]

    def test_missing_truth_excluded(self):
        report = calibration_eligibility(
            [rec(pred=2.2, gt=None)], now=100.5
        )
        assert report.n_eligible == 0
        assert "missing ground truth" in report.excluded[0][3]

    def test_invalid_truth_excluded(self):
        for bad_gt in (0.0, -1.0, float("nan")):
            report = calibration_eligibility(
                [rec(pred=2.2, gt=bad_gt)], now=100.5
            )
            assert report.n_eligible == 0, bad_gt

    def test_fit_actually_skips_ineligible(self):
        """Affine must fit ONLY on eligible rows - regression proof."""
        rows = [
            rec(frame=1, pred=1.1, gt=1.0),
            rec(frame=2, pred=2.2, gt=2.0),
            rec(frame=3, pred=3.3, gt=3.0),
            rec(frame=4, provenance="UNAVAILABLE", pred=None, gt=99.0),
            rec(frame=5, pred=4.4, gt=None),
            rec(frame=6, pred=float("inf"), gt=4.0),
        ]
        frozen = AffineCalibration().fit(rows)  # fit() filters internally
        # Fitted on the three clean rows: pred 2.2 -> exactly 2.0.
        assert frozen.transform(2.2) == pytest.approx(2.0, abs=1e-9)

    def test_planned_target_vs_independent_truth(self):
        """§11: 2.00 planned vs 1.97 truth is a VALID independent pair."""
        plan = TargetDistancePlan(targets_m=(2.0,), repetitions=1)
        protocol = ExperimentProtocol(
            plan=plan, scenes=("sim_contract",), experiment_id="p12"
        )
        c = protocol.start_capture("sim_contract", 2.0, 1)
        r = rec(frame=20, track=1, pred=2.2, gt=None)
        c.add_records([r])
        assert c.attach_ground_truth(
            {("sim_contract", 20, 1): 1.97}
        ) == 1
        c.complete()
        derived = c.derived_records()[0]
        # Target (slot key) and truth (record) remain distinct.
        assert c.target_distance_m == 2.0
        assert derived.ground_truth_distance_m == pytest.approx(1.97)
        assert derived.ground_truth_distance_m != c.target_distance_m
        report = calibration_eligibility([derived], now=100.0)
        assert report.n_eligible == 1

    def test_truth_generated_from_prediction_is_not_a_framework_path(self):
        """§11 case 4: no API derives truth from the predicted value.

        Structural proof: MeasurementRecord has no such constructor
        path and Capture.attach_ground_truth rejects derived-truth
        shortcuts - truth arrives only as an explicit operator value.
        """
        import inspect
        from calibration.capture import Capture
        src = inspect.getsource(Capture.attach_ground_truth)
        assert "float(values[key])" in src  # explicit operator value only
        # And a truth equal to the prediction is accepted ONLY as
        # recorded operator input (eligible, honestly labeled).
        report = calibration_eligibility(
            [rec(pred=2.0, gt=2.0)], now=100.5
        )
        assert report.n_eligible == 1  # equality alone is not a defect


# --------------------------------------------------------------
# 17. Split leakage (§12)
# --------------------------------------------------------------

class TestDatasetAcceptance:
    def test_accepts_clean_frozen_dataset(self):
        plan = TargetDistancePlan(targets_m=(1.0,), repetitions=1)
        protocol = ExperimentProtocol(
            plan=plan, scenes=("sim_contract",), experiment_id="p12"
        )
        c = protocol.start_capture("sim_contract", 1.0, 1)
        c.add_records([rec(frame=10, pred=1.1, gt=1.0)])
        c.complete()
        protocol.assign(c, CALIBRATION)
        protocol.freeze()
        acceptance = accept_dataset(
            [rec(frame=10, pred=1.1, gt=1.0)],
            [rec(frame=20, pred=1.1, gt=1.0)],
            protocol=protocol,
        )
        assert acceptance.accepted is True

    def test_overlap_raises_invalid_audit_state(self):
        shared = [rec(frame=10, pred=1.1, gt=1.0)]
        with pytest.raises(Exception) as excinfo:
            accept_dataset(shared, [rec(frame=10, pred=1.1, gt=1.0)])
        assert "INVALID-AUDIT-STATE" in str(excinfo.value)

    def test_unfrozen_protocol_rejected(self):
        plan = TargetDistancePlan(targets_m=(1.0,), repetitions=1)
        protocol = ExperimentProtocol(
            plan=plan, scenes=("sim_contract",), experiment_id="p12"
        )
        with pytest.raises(Exception) as excinfo:
            accept_dataset(
                [rec(frame=10, pred=1.1, gt=1.0)],
                [rec(frame=20, pred=1.1, gt=1.0)],
                protocol=protocol, require_frozen=True,
            )
        assert "not frozen" in str(excinfo.value)

    def test_stale_observations_reported_invalid_for_fit(self):
        acceptance = accept_dataset(
            [rec(frame=10, pred=1.1, gt=1.0, ts=90.0)],  # stale at now
            [rec(frame=20, pred=1.1, gt=1.0)],
            now=100.0, stale_after_s=2.0,
        )
        assert acceptance.accepted is False
        assert any("stale" in entry[4] for entry in
                   acceptance.invalid_for_fit)
        assert acceptance.summary["verdicts"]["calibration"]["STALE"] == 1

    def test_unavailable_reported_invalid_for_fit(self):
        acceptance = accept_dataset(
            [rec(frame=10, provenance="UNAVAILABLE", pred=None, gt=1.0)],
            [rec(frame=20, pred=1.1, gt=1.0)],
        )
        assert acceptance.accepted is False
        assert any("unavailable" in entry[4] for entry in
                   acceptance.invalid_for_fit)

    def test_missing_truth_reported_invalid_for_fit(self):
        acceptance = accept_dataset(
            [rec(frame=10, pred=1.1, gt=None)],
            [rec(frame=20, pred=1.1, gt=1.0)],
        )
        assert acceptance.accepted is False
        assert any("missing ground truth" in entry[4] for entry in
                   acceptance.invalid_for_fit)

    def test_missing_required_targets_block_acceptance(self):
        acceptance = accept_dataset(
            [rec(frame=10, pred=1.1, gt=1.0)],
            [rec(frame=20, pred=1.1, gt=1.0)],
            requested_targets_m=(1.0, 4.0),
        )
        assert acceptance.accepted is False
        assert acceptance.summary["missing_targets_m"] == [4.0]

    def test_inconsistent_provenance_is_invalid_audit_state(self):
        with pytest.raises(Exception) as excinfo:
            accept_dataset(
                [rec(frame=10, pred=1.1, gt=1.0, provenance="BOGUS")],
                [rec(frame=20, pred=1.1, gt=1.0)],
            )
        assert "INVALID-AUDIT-STATE" in str(excinfo.value)


# --------------------------------------------------------------
# 18-19. Provenance survives serialization / manifest (§9)
# --------------------------------------------------------------

class TestProvenanceSurvival:
    def test_provenance_survives_json_round_trip(self):
        for prov in ("MEASURED", "SIMULATED", "UNAVAILABLE", "STALE"):
            r = rec(provenance=prov,
                    pred=None if prov in ("UNAVAILABLE", "STALE") else 2.0)
            r2 = MeasurementRecord.parse_json(r.to_json())
            assert r2.distance_provenance == prov

    def test_provenance_survives_manifest_round_trip(self):
        plan = TargetDistancePlan(targets_m=(2.0,), repetitions=1)
        protocol = ExperimentProtocol(
            plan=plan, scenes=("sim_contract",), experiment_id="p12"
        )
        c = protocol.start_capture("sim_contract", 2.0, 1)
        r = rec(frame=20, track=1, pred=2.2, provenance="MEASURED")
        c.add_records([r])
        c.attach_ground_truth({("sim_contract", 20, 1): 1.97})
        c.complete()
        protocol.freeze()
        rehydrated = ExperimentProtocol.from_manifest(
            json.loads(json.dumps(protocol.manifest()))
        )
        # Truth audit (with its independent value) survived rehydration.
        assert rehydrated.find_capture("sim_contract", 2.0, 1) \
            .missing_truth() == 0

    def test_full_pipeline_provenance_flow(self):
        """provider -> fusion -> record -> capture -> derived ->
        calibration -> report: raw provenance never changes category."""
        # --- provider (measured-CONTRACT synthetic fixture) ---------
        provider = ScriptedDepthProvider(
            [{"person": ("MEASURED", 2.2)}], name="contract-fixture"
        )
        fusion = DistanceFusion(provider, clock=lambda: 100.0,
                                allow_simulated=True)
        detection_manager = DetectionManager()
        # --- fusion ---------------------------------------------------
        fusion.update()
        fused = fusion.fuse([det(bbox=(10, 10, 60, 160))], (400, 640))[0]
        assert fused.distance_provenance == "MEASURED"
        # --- record ---------------------------------------------------
        from navigation.navigator import NavigationDecision
        from navigation.navigation_types import NavigationAction
        from evaluation.recorder import MeasurementRecorder
        recorder = MeasurementRecorder(scene_id="sim_contract")
        record = recorder.capture_frame(
            frame_id=20, timestamp=100.0, detections=[fused],
            decision=NavigationDecision(NavigationAction.CONTINUE,
                                        "contract test"),
        )[0]
        assert record.distance_provenance == "MEASURED"
        # --- capture -> derived ---------------------------------------
        plan = TargetDistancePlan(targets_m=(2.0,), repetitions=1)
        protocol = ExperimentProtocol(
            plan=plan, scenes=("sim_contract",), experiment_id="p12"
        )
        c = protocol.start_capture("sim_contract", 2.0, 1)
        c.add_records([record])
        c.attach_ground_truth({("sim_contract", 20, None): 1.97})
        c.complete()
        derived = c.derived_records()[0]
        assert derived.distance_provenance == "MEASURED"
        assert derived.ground_truth_distance_m == pytest.approx(1.97)
        # --- calibration -----------------------------------------------
        fit_rows = [
            rec(frame=i, provenance="MEASURED", pred=1.0 + i / 10,
                gt=1.0) for i in range(1, 5)
        ]
        frozen = AffineCalibration().fit(fit_rows)
        obs = apply_calibration([derived], frozen)[0]
        assert obs.calibrated_provenance == "MEASURED_CALIBRATED"
        # --- report ----------------------------------------------------
        summary = measurement_quality_summary(
            [derived], calibrated_observations=[obs]
        )
        assert summary["by_provenance"]["MEASURED"] == 1
        assert summary["calibrated_by_provenance"]["MEASURED_CALIBRATED"] == 1


# --------------------------------------------------------------
# 20. Measurement quality summary (§13)
# --------------------------------------------------------------

class TestQualitySummary:
    def test_counts_all_categories(self):
        rows = [
            rec(frame=1, provenance="MEASURED", pred=1.1, gt=1.0),
            rec(frame=2, provenance="SIMULATED", pred=1.1, gt=1.0),
            rec(frame=3, provenance="UNAVAILABLE", pred=None, gt=None),
            rec(frame=4, provenance="STALE", pred=None, gt=None),
            rec(frame=5, provenance="MEASURED", pred=1.1, gt=None),
        ]
        frozen = AffineCalibration().fit(
            [rec(frame=i, provenance="MEASURED", pred=1.0 + i / 10, gt=1.0)
             for i in range(1, 4)]
        )
        obs = apply_calibration(rows, frozen)
        summary = measurement_quality_summary(rows, calibrated_observations=obs)
        assert summary["total"] == 5
        # valid = usable provenance + both values present (rows 1, 2;
        # row 5 is MEASURED+pred but gt=None -> invalid for accuracy).
        assert summary["valid"] == 2
        assert summary["invalid"] == 3
        assert summary["by_provenance"] == {
            "MEASURED": 2, "SIMULATED": 1, "UNAVAILABLE": 1, "STALE": 1,
        }
        assert summary["missing_truth"] == 3  # rows 3, 4, 5
        assert summary["truth_coverage"] == pytest.approx(2 / 5)
        assert summary["by_provenance"]["MEASURED"] == 2
        assert summary["calibrated_by_provenance"]["MEASURED_CALIBRATED"] >= 1
        assert summary["calibrated_by_provenance"][
            "SIMULATED_CALIBRATED"] >= 1
        assert summary["classification"] == (
            "MIXED PROVENANCES - split by provenance before comparing"
        )

    def test_measured_and_simulated_never_combined(self):
        rows = [rec(frame=1, provenance="MEASURED", pred=1.1, gt=1.0),
                rec(frame=2, provenance="SIMULATED", pred=1.1, gt=1.0)]
        summary = measurement_quality_summary(rows)
        # The summary reports counts, not a single accuracy claim.
        assert "mae" not in summary
        assert summary["note"].startswith("Descriptive counts only")

    def test_deterministic(self):
        rows = [rec(frame=1, provenance="MEASURED", pred=1.1, gt=1.0),
                rec(frame=2, provenance="SIMULATED", pred=1.1, gt=1.0)]
        a = measurement_quality_summary(rows)
        b = measurement_quality_summary(rows)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --------------------------------------------------------------
# 21-22. Two distinct gates (§14/§15)
# --------------------------------------------------------------

class TestSoftwareReadiness:
    def test_software_readiness_pass(self):
        result = software_calibration_readiness()
        assert result["software_calibration_readiness"] == "PASS"
        assert all(result["checks"].values()), result["failed_check_details"]
        assert result["schema"] == "sina-software-readiness/1"

    def test_hardware_gate_remains_separate_and_blocked(self):
        result = software_calibration_readiness()
        assert result["hardware_checkpoint"] == "BLOCKED"
        assert "NEVER implies hardware validation" in result["hardware_note"]
        assert result["classification"]["hardware"].startswith(
            "NOT VALIDATED"
        )
        assert result["classification"]["safety"] == "NOT VALIDATED"

    def test_readiness_check_list_complete(self):
        result = software_calibration_readiness()
        assert set(result["checks"]) == {
            "provider_contract_importable",
            "measured_provenance_supported",
            "calibrated_provenances_defined",
            "capture_protocol_available",
            "ground_truth_attachment_available",
            "split_enforcement_available",
            "manifest_persistence_available",
            "dataset_freeze_available",
            "report_generation_available",
            "raw_derived_separation_available",
        }

    def test_ready_report_does_not_unlock_hardware(self):
        """A PASS must not be readable as hardware approval."""
        result = software_calibration_readiness()
        text = json.dumps(result)
        assert "BLOCKED" in text
        assert "NOT VALIDATED" in text


# --------------------------------------------------------------
# 23. Synthetic cannot be relabeled as measured
# --------------------------------------------------------------

class TestNoRelabeling:
    def test_record_is_immutable(self):
        r = rec(provenance="SIMULATED", pred=2.0, gt=2.0)
        with pytest.raises(AttributeError):
            r.distance_provenance = "MEASURED"

    def test_taxonomy_has_no_conversion(self):
        """SIMULATED and MEASURED are distinct; calibrated labels are
        derived, and nothing in the contract module writes provenance."""
        assert "SIMULATED" in SOURCE_PROVENANCES
        assert "MEASURED" in SOURCE_PROVENANCES
        assert "SIMULATED" not in CALIBRATED_PROVENANCES
        assert "MEASURED" not in CALIBRATED_PROVENANCES
        assert set(SOURCE_PROVENANCES).isdisjoint(CALIBRATED_PROVENANCES)
        assert set(ALL_PROVENANCES) == (
            set(SOURCE_PROVENANCES) | set(CALIBRATED_PROVENANCES)
        )

    def test_provider_contract_fixture_is_not_hardware_claim(self):
        """The ('MEASURED', v) script spec emulates the provider contract
        for downstream testing only - it is still synthetic data."""
        provider = ScriptedDepthProvider(
            [{"person": ("MEASURED", 2.0)}], name="contract-fixture"
        )
        provider.update()
        m = provider.get_measurement((0, 0, 40, 120), (400, 640), "person",
                                     now=100.0)
        # The provenance string is honest about the CONTRACT being
        # tested; the module docstring records that the value is
        # synthetic and must never be presented as hardware-validated.
        assert m.provenance_str == "MEASURED"
        assert provider.name == "contract-fixture"
