"""
tests/test_evaluation_infra.py

OFFLINE MEASUREMENT & EVALUATION INFRASTRUCTURE TESTS (hardware-free).

Covers (milestone §14):

  - MeasurementRecord serialization/deserialization, missing fields,
    invalid values, CSV form, identity keys
  - metrics: exact MAE/RMSE/median/max/p95/relative on hand-computed
    sets, empty dataset, one-sample dataset, invalid measurements
    (None/UNAVAILABLE/STALE/gt<=0/predicted<=0) never silently become
    zero errors, invalid-measurement rate, provenance filtering,
    validation classification (SIMULATION vs MEASURED vs MIXED)
  - calibration/validation separation (overlap detection, assert)
  - ScriptedDepthProvider failure injection (unavailable, stale,
    old-timestamp, measured/simulated specs, script advance/hold)
  - ReplayRunner: determinism, all §12 failure scenarios through the
    REAL pipeline (fusion -> motion -> risk -> temporal navigation)
  - MeasurementRecorder capture + ground-truth joining
  - EventLogger structured trace (memory + file)
  - JSONL/summary file I/O round-trips

No OAK-D, no DepthAI import, no network, no GPU.
"""

import json

import pytest

from evaluation.record import (
    MeasurementRecord,
    SCHEMA_VERSION,
    VALID_PROVENANCES,
)
from evaluation.metrics import (
    evaluate,
    is_valid_record,
    validation_class,
    check_overlap,
    assert_disjoint,
)
from evaluation.report import read_jsonl, write_jsonl, write_summary_json
from evaluation.injection import ScriptedDepthProvider
from evaluation.recorder import MeasurementRecorder
from evaluation.event_log import EventLogger
from evaluation.replay import ReplayRunner, ReplayConfig, ScriptedFrame
from depth.provider import DistanceProvenance, DepthMeasurement
from vision.object_detector import DetectedObject, BoundingBox
from navigation.navigation_types import NavigationAction
from navigation.navigator import NavigationDecision


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


def sframe(detections, distances):
    return ScriptedFrame(detections=detections, distances=distances)


PERSON_CENTER = (280, 150, 360, 320)   # center_x=320 -> CENTER in 640 px


# ==========================================================
# MeasurementRecord schema
# ==========================================================

class TestRecordSchema:
    def test_roundtrip_dict(self):
        rec = make_record(track_id=3, ground_truth_distance_m=2.05)
        assert MeasurementRecord.from_dict(rec.to_dict()) == rec

    def test_roundtrip_json(self):
        rec = make_record(predicted_distance_m=None,
                          distance_provenance="UNAVAILABLE")
        parsed = MeasurementRecord.from_dict(json.loads(rec.to_json()))
        assert parsed == rec

    def test_ground_truth_none_is_preserved_not_zero(self):
        rec = make_record(ground_truth_distance_m=None)
        parsed = MeasurementRecord.from_dict(rec.to_dict())
        assert parsed.ground_truth_distance_m is None

    def test_missing_required_field_raises(self):
        d = make_record().to_dict()
        del d["predicted_distance_m"]
        with pytest.raises(KeyError):
            MeasurementRecord.from_dict(d)

    def test_newer_schema_rejected(self):
        d = make_record().to_dict()
        d["schema_version"] = SCHEMA_VERSION + 1
        with pytest.raises(ValueError):
            MeasurementRecord.from_dict(d)

    def test_unknown_provenance_rejected(self):
        d = make_record().to_dict()
        d["distance_provenance"] = "GUESSED"
        with pytest.raises(ValueError):
            MeasurementRecord.from_dict(d)

    def test_bad_bbox_rejected(self):
        d = make_record().to_dict()
        d["bbox"] = [1, 2, 3]
        with pytest.raises(ValueError):
            MeasurementRecord.from_dict(d)

    def test_csv_header_matches_fields(self):
        header = MeasurementRecord.csv_header()
        for name in ("scene_id", "frame_id", "track_id", "bbox", "region",
                     "predicted_distance_m", "distance_provenance",
                     "ground_truth_distance_m"):
            assert name in header

    def test_csv_row_none_empty_and_bbox_semicolons(self):
        rec = make_record(track_id=None, ground_truth_distance_m=None)
        row = rec.to_csv_row()
        cells = row.split(",")
        assert cells[4] == ""            # track_id None -> empty
        assert cells[12] == ""           # ground truth None -> empty
        assert cells[7] == "100;100;200;300"

    def test_identity_key(self):
        rec = make_record(scene_id="s1", frame_id=7, track_id=2)
        assert rec.identity_key() == ("s1", 7, 2)


# ==========================================================
# Metrics
# ==========================================================

class TestMetrics:
    def test_perfect_predictions_zero_error(self):
        recs = [make_record(frame_id=i, predicted_distance_m=2.0,
                            ground_truth_distance_m=2.0) for i in range(5)]
        s = evaluate(recs)
        assert s.n_valid == 5 and s.n_invalid == 0
        assert s.mae_m == 0.0 and s.rmse_m == 0.0
        assert s.max_abs_error_m == 0.0
        assert s.invalid_rate == 0.0

    def test_hand_computed_statistics(self):
        recs = [
            make_record(frame_id=0, predicted_distance_m=1.0,
                        ground_truth_distance_m=1.1),
            make_record(frame_id=1, predicted_distance_m=2.0,
                        ground_truth_distance_m=1.8),
            make_record(frame_id=2, predicted_distance_m=3.0,
                        ground_truth_distance_m=3.3),
        ]
        s = evaluate(recs)
        assert s.mae_m == pytest.approx(0.2)
        # RMSE = sqrt(mean([0.1^2, 0.2^2, 0.3^2]))
        assert s.rmse_m == pytest.approx((0.14 / 3) ** 0.5)
        assert s.median_abs_error_m == pytest.approx(0.2)
        assert s.max_abs_error_m == pytest.approx(0.3)
        assert s.median_relative_error == pytest.approx(
            0.1 / 1.1)  # median of (0.0909, 0.1111, 0.0909)
        assert s.p95_abs_error_m == pytest.approx(0.29)

    def test_empty_dataset(self):
        s = evaluate([])
        assert s.n_total == 0 and s.n_valid == 0
        assert s.mae_m is None and s.rmse_m is None
        assert s.validation_class == "NO USABLE MEASUREMENTS"
        assert s.invalid_rate == 0.0

    def test_one_sample_dataset(self):
        s = evaluate([make_record(predicted_distance_m=1.5,
                                  ground_truth_distance_m=1.0)])
        assert s.mae_m == pytest.approx(0.5)
        assert s.rmse_m == pytest.approx(0.5)
        assert s.max_abs_error_m == pytest.approx(0.5)

    def test_unavailable_and_stale_counted_invalid_not_zero(self):
        recs = [
            make_record(frame_id=0, predicted_distance_m=2.0),        # valid
            make_record(frame_id=1, predicted_distance_m=2.0),        # valid
            make_record(frame_id=2, predicted_distance_m=None,
                        distance_provenance="UNAVAILABLE"),
            make_record(frame_id=3, predicted_distance_m=None,
                        distance_provenance="STALE"),
            make_record(frame_id=4, predicted_distance_m=2.0,
                        ground_truth_distance_m=None),
        ]
        s = evaluate(recs)
        assert s.n_total == 5
        assert s.n_valid == 2
        assert s.n_invalid == 3
        assert s.invalid_rate == pytest.approx(0.6)
        # accuracy uses ONLY the valid rows
        assert s.mae_m == pytest.approx(0.0)

    def test_unavailable_with_number_is_still_invalid(self):
        """Non-usable provenance never enters accuracy, even with a value."""
        recs = [make_record(predicted_distance_m=2.0,
                            distance_provenance="UNAVAILABLE")]
        s = evaluate(recs)
        assert s.n_valid == 0 and s.mae_m is None

    def test_zero_and_negative_ground_truth_invalid(self):
        assert not is_valid_record(make_record(ground_truth_distance_m=0.0))
        assert not is_valid_record(make_record(ground_truth_distance_m=-1.0))
        assert not is_valid_record(make_record(predicted_distance_m=0.0))

    def test_small_ground_truth_relative_error(self):
        s = evaluate([make_record(predicted_distance_m=0.6,
                                  ground_truth_distance_m=0.5)])
        assert s.median_relative_error == pytest.approx(0.2)

    def test_provenance_filter_measured_only(self):
        recs = [
            make_record(frame_id=0, distance_provenance="SIMULATED",
                        predicted_distance_m=2.0,
                        ground_truth_distance_m=1.0),
            make_record(frame_id=1, distance_provenance="MEASURED",
                        predicted_distance_m=2.0,
                        ground_truth_distance_m=2.0),
        ]
        s = evaluate(recs, provenances=("MEASURED",))
        assert s.n_total == 1
        assert s.provenances == ("MEASURED",)
        assert s.mae_m == pytest.approx(0.0)

    def test_mixed_provenances_visible_not_silent(self):
        recs = [
            make_record(distance_provenance="SIMULATED"),
            make_record(distance_provenance="MEASURED"),
        ]
        s = evaluate(recs)
        assert set(s.provenances) == {"MEASURED", "SIMULATED"}
        assert s.validation_class.startswith("MIXED")

    def test_validation_class_simulation_only(self):
        assert validation_class(["SIMULATED"]) == "SIMULATION-VALIDATED"

    def test_validation_class_measured_only(self):
        assert validation_class(["MEASURED"]).startswith(
            "MEASUREMENT-VALIDATED")

    def test_validation_class_no_usable(self):
        assert validation_class(["UNAVAILABLE"]) == "NO USABLE MEASUREMENTS"


# ==========================================================
# Calibration / validation separation
# ==========================================================

class TestCalibrationValidationSeparation:
    def test_disjoint_sets_pass(self):
        calib = [make_record(scene_id="c", frame_id=i) for i in range(3)]
        val = [make_record(scene_id="v", frame_id=i) for i in range(3)]
        report = assert_disjoint(calib, val)
        assert report.is_clean

    def test_exact_overlap_raises(self):
        calib = [make_record(scene_id="s", frame_id=1, track_id=1)]
        val = [make_record(scene_id="s", frame_id=1, track_id=1)]
        with pytest.raises(ValueError, match="overlap"):
            assert_disjoint(calib, val)

    def test_shared_scene_without_shared_observation_is_reported(self):
        calib = [make_record(scene_id="s", frame_id=1, track_id=1)]
        val = [make_record(scene_id="s", frame_id=1, track_id=2)]
        report = check_overlap(calib, val)
        assert report.is_clean                       # no identical triples
        assert report.shared_scene_ids == ("s",)     # but scene is shared

    def test_same_scene_different_frames_clean(self):
        calib = [make_record(scene_id="s", frame_id=i) for i in range(2)]
        val = [make_record(scene_id="s", frame_id=i + 10) for i in range(2)]
        assert assert_disjoint(calib, val).is_clean


# ==========================================================
# ScriptedDepthProvider (failure injection)
# ==========================================================

class TestScriptedDepthProvider:
    BBOX = (0, 0, 10, 10)
    SHAPE = (400, 640)

    def _meas(self, provider, label="person", now=10.0):
        return provider.get_measurement(self.BBOX, self.SHAPE,
                                        label=label, now=now)

    def test_float_spec_is_simulated(self):
        p = ScriptedDepthProvider([{"person": 2.5}])
        m = self._meas(p)
        assert m.distance_m == 2.5
        assert m.provenance is DistanceProvenance.SIMULATED

    def test_none_spec_is_unavailable(self):
        p = ScriptedDepthProvider([{"person": None}])
        m = self._meas(p)
        assert m.distance_m is None
        assert m.provenance is DistanceProvenance.UNAVAILABLE

    def test_measured_spec_carries_measured_provenance(self):
        p = ScriptedDepthProvider([{"person": ("MEASURED", 2.5)}])
        m = self._meas(p)
        assert m.provenance is DistanceProvenance.MEASURED
        assert m.distance_m == 2.5

    def test_stale_spec_is_stream_quiet(self):
        """STALE follows the real provider contract: no distance value."""
        p = ScriptedDepthProvider([{"person": ("STALE", 2.5)}])
        m = self._meas(p)
        assert m.provenance is DistanceProvenance.STALE
        assert m.distance_m is None

    def test_old_spec_timestamp_is_in_past(self):
        p = ScriptedDepthProvider([{"person": ("OLD", 2.5)}],
                                  stale_age_s=5.0)
        m = self._meas(p, now=100.0)
        assert m.timestamp == pytest.approx(95.0)

    def test_unknown_label_unavailable(self):
        p = ScriptedDepthProvider([{"person": 2.0}])
        m = self._meas(p, label="chair")
        assert m.provenance is DistanceProvenance.UNAVAILABLE

    def test_script_advances_per_update_and_holds_last(self):
        """Contract: pre-first-update queries clamp to script frame 0;
        each update() advances one frame; beyond the end the last is held.
        (Replay calls update() once per frame before measuring, so script
        frame N is served during replay frame N.)"""
        p = ScriptedDepthProvider([{"person": 1.0}, {"person": 2.0}])
        assert self._meas(p).distance_m == 1.0   # clamp to frame 0
        p.update()
        assert self._meas(p).distance_m == 1.0   # frame 0
        p.update()
        assert self._meas(p).distance_m == 2.0   # frame 1
        p.update()
        assert self._meas(p).distance_m == 2.0   # held last

    def test_empty_script_rejected(self):
        with pytest.raises(ValueError):
            ScriptedDepthProvider([])


# ==========================================================
# Replay (real pipeline, scripted scenes)
# ==========================================================

class TestReplay:
    def _run(self, frames, scene_id="test"):
        return ReplayRunner(ReplayConfig(scene_id=scene_id)).run(frames)

    def test_determinism(self):
        frames = [
            sframe([("person", PERSON_CENTER, 0.9)], {"person": 2.0}),
            sframe([("person", PERSON_CENTER, 0.9)], {"person": 1.9}),
            sframe([("person", PERSON_CENTER, 0.9)], {"person": 1.8}),
        ]
        r1 = self._run(frames)
        r2 = self._run(frames)
        assert r1.records == r2.records
        assert r1.events == r2.events
        # NavigationDecision has no __eq__: compare by value.
        assert [(d.action.name, d.reason, d.priority) for d in r1.decisions] \
            == [(d.action.name, d.reason, d.priority) for d in r2.decisions]

    def test_far_person_continues(self):
        frames = [sframe([("person", PERSON_CENTER, 0.9)], {"person": 4.5})
                  for _ in range(3)]
        r = self._run(frames)
        assert all(d.action is NavigationAction.CONTINUE for d in r.decisions)

    def test_critical_distance_stops(self):
        frames = [sframe([("person", PERSON_CENTER, 0.9)], {"person": 1.0})
                  for _ in range(3)]
        r = self._run(frames)
        assert r.decisions[-1].action is NavigationAction.STOP

    def test_sudden_jump_to_near_stops_immediately(self):
        frames = [
            sframe([("person", PERSON_CENTER, 0.9)], {"person": 4.0}),
            sframe([("person", PERSON_CENTER, 0.9)], {"person": 0.5}),
        ]
        r = self._run(frames)
        assert r.decisions[1].action is NavigationAction.STOP

    def test_unavailable_distance_falls_back_to_spatial(self):
        """Person center is a spatial hazard -> legacy STOP despite no depth."""
        frames = [sframe([("person", PERSON_CENTER, 0.9)], {"person": None})
                  for _ in range(3)]
        r = self._run(frames)
        assert r.decisions[-1].action is NavigationAction.STOP
        assert all(rec.distance_provenance == "UNAVAILABLE"
                   for rec in r.records)
        assert all(rec.predicted_distance_m is None for rec in r.records)

    def test_stale_depth_rejected_not_reused(self):
        frames = [sframe([("person", PERSON_CENTER, 0.9)],
                         {"person": ("STALE", 1.0)})
                  for _ in range(3)]
        r = self._run(frames)
        assert all(rec.distance_provenance == "STALE" for rec in r.records)
        assert all(rec.predicted_distance_m is None for rec in r.records)

    def test_old_timestamp_demoted_to_stale_by_fusion(self):
        """('OLD', v) is SIMULATED with an old timestamp: fusion must
        demote it to STALE and withhold the value (defense in depth)."""
        frames = [sframe([("person", PERSON_CENTER, 0.9)],
                         {"person": ("OLD", 1.0)})
                  for _ in range(2)]
        r = self._run(frames)
        assert all(rec.distance_provenance == "STALE" for rec in r.records)
        assert all(rec.predicted_distance_m is None for rec in r.records)

    def test_simulated_to_measured_transition(self):
        frames = [
            sframe([("person", PERSON_CENTER, 0.9)], {"person": 2.0}),
            sframe([("person", PERSON_CENTER, 0.9)], {"person": 2.0}),
            sframe([("person", PERSON_CENTER, 0.9)], {"person": 2.0}),
            sframe([("person", PERSON_CENTER, 0.9)],
                   {"person": ("MEASURED", 2.0)}),
            sframe([("person", PERSON_CENTER, 0.9)],
                   {"person": ("MEASURED", 2.0)}),
        ]
        r = self._run(frames)
        provs = [rec.distance_provenance for rec in r.records
                 if rec.frame_id >= 3]
        assert provs and all(p == "MEASURED" for p in provs)
        early = [rec.distance_provenance for rec in r.records
                 if rec.frame_id < 3]
        assert all(p == "SIMULATED" for p in early)

    def test_measured_to_unavailable_transition_no_crash(self):
        frames = [
            sframe([("person", PERSON_CENTER, 0.9)],
                   {"person": ("MEASURED", 2.0)}),
            sframe([("person", PERSON_CENTER, 0.9)],
                   {"person": ("MEASURED", 2.0)}),
            sframe([("person", PERSON_CENTER, 0.9)], {"person": None}),
            sframe([("person", PERSON_CENTER, 0.9)], {"person": None}),
        ]
        r = self._run(frames)
        assert len(r.decisions) == 4
        late = [rec for rec in r.records if rec.frame_id >= 2]
        assert all(rec.distance_provenance == "UNAVAILABLE"
                   for rec in late)

    def test_disappearing_and_returning_track_keeps_id(self):
        present = sframe([("person", PERSON_CENTER, 0.9)], {"person": 2.0})
        empty = sframe([], {})
        frames = [present, present, present, empty, empty, present]
        r = self._run(frames)
        ids = [rec.track_id for rec in r.records]
        assert ids[0] == 1 and ids[-1] == 1   # same track after return
        assert len(r.decisions) == 6          # empty frames decided fine

    def test_approaching_person_escalates_and_stays_stopped(self):
        # 4.0 m decreasing 0.06 m/frame (~0.9 m/s) -> STOP by ~frame 47.
        frames = [
            sframe([("person", PERSON_CENTER, 0.9)],
                   {"person": round(4.0 - 0.06 * i, 3)})
            for i in range(50)
        ]
        r = self._run(frames)
        actions = [d.action for d in r.decisions]
        assert actions[0] is NavigationAction.CONTINUE
        assert NavigationAction.SLOW_DOWN in actions or \
            NavigationAction.STOP in actions
        assert actions[-1] is NavigationAction.STOP
        # once STOP appears it never relaxes within the approach
        first_stop = actions.index(NavigationAction.STOP)
        assert all(a is NavigationAction.STOP for a in actions[first_stop:])

    def test_receding_person_eventually_deesculates(self):
        frames = [
            sframe([("person", PERSON_CENTER, 0.9)],
                   {"person": round(1.0 + 0.3 * i, 3)})
            for i in range(14)
        ]
        r = self._run(frames)
        actions = [d.action for d in r.decisions]
        assert actions[0] is NavigationAction.STOP
        assert actions[-1] is not NavigationAction.STOP

    def test_low_risk_object_never_dilutes_critical_scene(self):
        bottle_left = (50, 200, 90, 240)   # LEFT region
        frames = [
            sframe(
                [("person", PERSON_CENTER, 0.9),
                 ("bottle", bottle_left, 0.8)],
                {"person": 1.0, "bottle": 5.0},
            )
            for _ in range(3)
        ]
        r = self._run(frames)
        assert r.decisions[-1].action is NavigationAction.STOP
        labels = {rec.label for rec in r.records}
        assert labels == {"person", "bottle"}

    def test_records_carry_track_and_provenance_verbatim(self):
        frames = [sframe([("chair", PERSON_CENTER, 0.8)], {"chair": 2.5})
                  for _ in range(2)]
        r = self._run(frames)
        rec = r.records[0]
        assert rec.label == "chair"
        assert rec.distance_provenance == "SIMULATED"
        assert rec.distance_source == "scripted:test"
        assert rec.track_id == 1
        assert rec.ground_truth_distance_m is None   # never fabricated
        assert rec.region == "CENTER"

    def test_empty_frames_only(self):
        frames = [sframe([], {}) for _ in range(2)]
        r = self._run(frames)
        assert r.records == []
        assert all(d.action is NavigationAction.CONTINUE
                   for d in r.decisions)

    def test_events_contain_annotated_fields(self):
        frames = [sframe([("person", PERSON_CENTER, 0.9)], {"person": 1.0})]
        r = self._run(frames)
        obj = r.events[0]["objects"][0]
        for key in ("track_id", "label", "region", "confidence", "distance",
                    "distance_provenance", "motion_state", "closing_rate_mps",
                    "risk_level", "risk_ttc_s"):
            assert key in obj
        assert r.events[0]["action"] == "STOP"


# ==========================================================
# MeasurementRecorder + EventLogger
# ==========================================================

def _live_frame_objects():
    det = DetectedObject(
        label="person", confidence=0.9, bbox=BoundingBox(10, 10, 50, 100),
        center_x=30, center_y=55, region="CENTER", width=40, height=90,
        area=3600, priority=100, distance=2.5,
        distance_provenance="MEASURED", distance_source="oak_stereo",
        track_id=4, motion_state="APPROACHING", closing_rate_mps=0.4,
        motion_provenance="MEASURED", risk_level="HIGH",
        risk_reasons=["distance 2.5m"], risk_ttc_s=6.2,
        risk_provenance="MEASURED",
    )
    decision = NavigationDecision(
        action=NavigationAction.SLOW_DOWN, reason="Person ahead at 2.5m",
        priority=100, trigger=det)
    return [det], decision


class TestRecorder:
    def test_capture_copies_provenance_verbatim(self):
        rec = MeasurementRecorder(scene_id="live")
        dets, decision = _live_frame_objects()
        records = rec.capture_frame(0, 10.0, dets, decision)
        assert len(records) == 1
        r = records[0]
        assert r.scene_id == "live"
        assert r.track_id == 4
        assert r.predicted_distance_m == 2.5
        assert r.distance_provenance == "MEASURED"
        assert r.distance_source == "oak_stereo"
        assert r.ground_truth_distance_m is None

    def test_attach_truth_joins_by_identity(self):
        rec = MeasurementRecorder(scene_id="live")
        dets, decision = _live_frame_objects()
        rec.capture_frame(3, 10.0, dets, decision)
        joined = rec.attach_truth({("live", 3, 4): 2.61})
        assert joined == 1
        assert rec.records[0].ground_truth_distance_m == 2.61

    def test_attach_truth_leaves_unmatched_none(self):
        rec = MeasurementRecorder(scene_id="live")
        dets, decision = _live_frame_objects()
        rec.capture_frame(0, 10.0, dets, decision)
        rec.attach_truth({("live", 99, 4): 1.0})   # wrong frame
        assert rec.records[0].ground_truth_distance_m is None

    def test_event_log_memory_trace_fields(self):
        rec = MeasurementRecorder(scene_id="live")
        dets, decision = _live_frame_objects()
        rec.capture_frame(0, 10.0, dets, decision)
        event = rec.events.parsed()[0]
        assert event["action"] == "SLOW_DOWN"
        obj = event["objects"][0]
        assert obj["risk_level"] == "HIGH"
        assert obj["motion_state"] == "APPROACHING"
        assert obj["distance_provenance"] == "MEASURED"


class TestEventLoggerFile:
    def test_file_sink_writes_jsonl(self, tmp_path):
        path = tmp_path / "trace" / "events.jsonl"
        logger = EventLogger(sink=path)
        dets, decision = _live_frame_objects()
        logger.log_frame(0, 1.0, dets, decision)
        logger.close()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["frame"] == 0 and parsed["action"] == "SLOW_DOWN"


# ==========================================================
# File I/O
# ==========================================================

class TestReportIO:
    def test_jsonl_roundtrip(self, tmp_path):
        path = tmp_path / "rec.jsonl"
        recs = [make_record(frame_id=i, track_id=i) for i in range(3)]
        assert write_jsonl(path, recs) == 3
        assert read_jsonl(path) == recs

    def test_jsonl_append(self, tmp_path):
        path = tmp_path / "rec.jsonl"
        write_jsonl(path, [make_record(frame_id=0)])
        write_jsonl(path, [make_record(frame_id=1)], append=True)
        assert len(read_jsonl(path)) == 2

    def test_jsonl_skips_blank_lines(self, tmp_path):
        path = tmp_path / "rec.jsonl"
        path.write_text(
            make_record(frame_id=0).to_json() + "\n\n"
            + make_record(frame_id=1).to_json() + "\n",
            encoding="utf-8")
        assert len(read_jsonl(path)) == 2

    def test_summary_json_roundtrip(self, tmp_path):
        path = tmp_path / "summary.json"
        s = evaluate([make_record(predicted_distance_m=1.5,
                                  ground_truth_distance_m=1.0)])
        write_summary_json(path, s)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["mae_m"] == pytest.approx(0.5)
        assert data["validation_class"] == "SIMULATION-VALIDATED"
