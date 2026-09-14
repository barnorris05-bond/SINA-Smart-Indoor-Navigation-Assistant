"""
calibration/contract.py

PHYSICAL MEASUREMENT CONTRACT & CALIBRATION READINESS (Phase 12).

SOFTWARE-ONLY. This module hardens the boundary between a future REAL
measured distance and the existing evaluation/capture/calibration
stack. It keeps three separate questions separate:

A. MEASUREMENT VALIDITY   - is this observation's numeric content
   usable at all? (finite, positive, fresh, known identity)
B. MEASUREMENT PROVENANCE - where did the value come from?
   ("MEASURED" | "SIMULATED" | "UNAVAILABLE" | "STALE" - the project
   vocabulary from depth.provider.DistanceProvenance, reused verbatim)
C. CALIBRATION PROVENANCE - what happened to the value afterwards?
   ("MEASURED_CALIBRATED" | "SIMULATED_CALIBRATED" - derived ONLY from
   the source provenance by a calibration transform; never hand-relabeled)

Core invariants (exhaustively tested in tests/test_measurement_contract.py):

- SIMULATED can never become MEASURED through any path in this repo.
- UNAVAILABLE stays unavailable; STALE can never enter a fresh dataset
  as a current measurement (freshness is a REAL contract requirement
  evidenced by timestamps - mirroring DistanceFusion's staleness gate).
- Calibrated outputs never overwrite raw records: they are separate
  derived products (CalibratedObservation) with explicit labels.
- Invalid measurements (non-finite, zero, negative, missing truth)
  never enter calibration fitting.
- Ground truth must be INDEPENDENTLY supplied; there is no path that
  derives truth from the predicted value.

Hardware availability is deliberately OUT OF SCOPE: the software
readiness check below never answers "is the OAK-D working?" - that
gate (Phase 9 Checkpoint A) remains BLOCKED and separate.
"""

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from evaluation.metrics import assert_disjoint
from evaluation.record import MeasurementRecord, VALID_PROVENANCES

from calibration.experiment import (
    CalibratedObservation,
    CalibrationExperiment,
    apply_calibration,
)
from calibration.models import (
    CALIBRATED_PROVENANCE,
    CALIBRATED_PROVENANCE_SIMULATED,
    AffineCalibration,
    IdentityCalibration,
    usable_provenance_for_scoring,
)
from calibration.plan import TargetDistancePlan
from calibration.protocol import ExperimentProtocol

# --------------------------------------------------------------
# The full provenance taxonomy (source + calibrated vocabulary)
# --------------------------------------------------------------

SOURCE_PROVENANCES: Tuple[str, ...] = tuple(VALID_PROVENANCES)
CALIBRATED_PROVENANCES: Tuple[str, ...] = (
    CALIBRATED_PROVENANCE,
    CALIBRATED_PROVENANCE_SIMULATED,
)
ALL_PROVENANCES: Tuple[str, ...] = SOURCE_PROVENANCES + CALIBRATED_PROVENANCES

#: Provenances that may legally enter a CALIBRATION FIT.
FIT_ELIGIBLE_PROVENANCES: Tuple[str, ...] = ("MEASURED", "SIMULATED")


# --------------------------------------------------------------
# A. Measurement validity - the observation-level verdict
# --------------------------------------------------------------

# Explicit verdicts (never silent clamping; every rejection states why).
VERDICT_VALID = "VALID"
VERDICT_STALE = "STALE"
VERDICT_UNAVAILABLE = "UNAVAILABLE"
VERDICT_INVALID = "INVALID"


@dataclass(frozen=True)
class ObservationVerdict:
    """
    One observation's contract verdict (Phase 12 §3/§5).

    classification is one of VALID / STALE / UNAVAILABLE / INVALID and
    reasons lists the exact violated conditions (empty when VALID).
    The verdict NEVER changes the observation - it only describes it.
    """

    classification: str
    reasons: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.classification == VERDICT_VALID

    def to_dict(self) -> Dict[str, object]:
        return {
            "classification": self.classification,
            "reasons": list(self.reasons),
        }


def check_observation(
    record: MeasurementRecord,
    now: Optional[float] = None,
    stale_after_s: float = 2.0,
) -> ObservationVerdict:
    """
    MEASURED-OBSERVATION CONTRACT (Phase 12 §3).

    An observation may legitimately carry a usable provenance
    (MEASURED / SIMULATED) only when ALL of the following hold -
    conditions already established by the codebase, restated as one
    contract:

    - distance present and FINITE (non-finite is INVALID, never clamped)
    - distance strictly POSITIVE (zero/negative is INVALID: a physical
      distance cannot be 0 m)
    - provenance inside the six-value taxonomy (source + calibrated)
    - UNAVAILABLE/STALE provenances carry NO value (a stale value must
      never masquerade as current - the DepthMeasurement contract)
    - finite, non-negative timestamp; a FUTURE timestamp (relative to
      `now`, when supplied) is INVALID - clocks never run backwards
    - FRESHNESS: a usable value whose timestamp is older than
      stale_after_s (default mirrors config MEASUREMENT_STALE_S) is
      classified STALE - stale data can never enter a fresh dataset
      as a current measurement
    - an observation identity: non-empty scene, non-negative frame

    now=None disables the freshness/future checks (caller opted out);
    production callers pass the injectable clock.
    """
    reasons: List[str] = []
    prov = record.distance_provenance

    # ---- Provenance vocabulary ------------------------------------
    if prov not in ALL_PROVENANCES:
        return ObservationVerdict(
            VERDICT_INVALID, (f"unknown provenance {prov!r}",)
        )

    # ---- Availability states (honest no-measurement) ---------------
    if record.predicted_distance_m is None:
        if prov in ("UNAVAILABLE", "STALE"):
            return ObservationVerdict(prov, ())
        return ObservationVerdict(
            VERDICT_INVALID, ("missing predicted distance",)
        )

    # A stale/unavailable label must never carry a value.
    if prov in ("UNAVAILABLE", "STALE"):
        return ObservationVerdict(
            VERDICT_INVALID,
            (f"{prov} provenance cannot carry a value "
             f"({record.predicted_distance_m!r})",),
        )

    # ---- Numeric content -------------------------------------------
    value = record.predicted_distance_m
    if not math.isfinite(value):
        reasons.append("non-finite predicted distance")
    elif value <= 0:
        reasons.append(
            f"non-positive predicted distance ({value}); a physical "
            "distance is strictly positive"
        )

    # ---- Timestamp contract -----------------------------------------
    ts = record.timestamp
    if not math.isfinite(ts):
        reasons.append("non-finite timestamp")
    elif ts < 0:
        reasons.append(f"negative timestamp ({ts})")
    elif now is not None and math.isfinite(now):
        if ts > now:
            reasons.append(f"future timestamp ({ts} > now {now})")
        elif (now - ts) > stale_after_s:
            # Freshness is REAL: old values must not masquerade as
            # current measurements (DistanceFusion's defense in depth,
            # restated at the dataset contract level).
            reasons.append(
                f"stale: age {now - ts:.3f}s > staleness window "
                f"{stale_after_s:.3f}s"
            )

    # ---- Observation identity ---------------------------------------
    if not record.scene_id:
        reasons.append("missing scene identity")
    if record.frame_id < 0:
        reasons.append(f"negative frame id ({record.frame_id})")

    if reasons:
        # A stale age is reported as STALE (not INVALID) when that is
        # the ONLY problem: the value itself was well-formed.
        if all(r.startswith("stale:") for r in reasons):
            return ObservationVerdict(VERDICT_STALE, tuple(reasons))
        return ObservationVerdict(VERDICT_INVALID, tuple(reasons))
    return ObservationVerdict(VERDICT_VALID, ())


# --------------------------------------------------------------
# Calibration eligibility (Phase 12 §8/§11)
# --------------------------------------------------------------

@dataclass(frozen=True)
class EligibilityReport:
    """
    Which records may legally enter a CALIBRATION FIT, and why the
    others may not. AUDIT-PRESENT BUT INVALID-FOR-FIT records are
    listed per reason; the caller decides what to do - nothing is
    silently dropped (Phase 12 §12).
    """

    eligible: Tuple[MeasurementRecord, ...]
    excluded: Tuple[Tuple[str, int, Optional[int], str], ...]
    n_total: int

    @property
    def n_eligible(self) -> int:
        return len(self.eligible)

    @property
    def n_excluded(self) -> int:
        return len(self.excluded)

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_total": self.n_total,
            "n_eligible": len(self.eligible),
            "n_excluded": len(self.excluded),
            "excluded": [
                {"scene_id": s, "frame_id": f, "track_id": t, "reason": r}
                for s, f, t, r in self.excluded
            ],
        }


def calibration_eligibility(
    records: Sequence[MeasurementRecord],
    now: Optional[float] = None,
    stale_after_s: float = 2.0,
) -> EligibilityReport:
    """
    FIT eligibility gate (Phase 12 §8): a record may enter calibration
    fitting only when it pairs a USABLE predicted measurement with an
    INDEPENDENT ground truth. Explicitly rejected:

    - UNAVAILABLE / STALE provenances (no usable value)
    - missing or non-positive ground truth (no independent reference)
    - non-finite / non-positive predicted values
    - non-finite / negative / future / stale timestamps
    - provenance outside the six-value taxonomy

    Ground truth is never inferred from the predicted value: this
    module has no path that writes ground_truth_distance_m, and the
    capture layer's truth attachment is operator-supplied and audited.
    A truth equal to the prediction is accepted only as recorded
    operator input (case 3 of §11).
    """
    eligible: List[MeasurementRecord] = []
    excluded: List[Tuple[str, int, Optional[int], str]] = []

    for rec in records:
        verdict = check_observation(rec, now=now, stale_after_s=stale_after_s)
        key = (rec.scene_id, rec.frame_id, rec.track_id)
        if verdict.classification == VERDICT_STALE:
            excluded.append(key + ("stale: cannot enter a fresh fit",))
            continue
        if verdict.classification == VERDICT_UNAVAILABLE:
            excluded.append(key + ("unavailable: no usable value",))
            continue
        if verdict.classification == VERDICT_INVALID:
            excluded.append(key + ("; ".join(verdict.reasons),))
            continue
        # VALID so far - now the independent-truth requirement.
        gt = rec.ground_truth_distance_m
        if gt is None:
            excluded.append(key + ("missing ground truth",))
            continue
        if not math.isfinite(gt) or gt <= 0:
            excluded.append(key + (f"invalid ground truth ({gt})",))
            continue
        eligible.append(rec)

    return EligibilityReport(
        eligible=tuple(eligible),
        excluded=tuple(excluded),
        n_total=len(records),
    )


# --------------------------------------------------------------
# Provenance invariants A-G as executable checks (Phase 12 §4)
# --------------------------------------------------------------

def provenance_invariant_report(
    records: Sequence[MeasurementRecord],
    calibrated: Optional[Sequence[CalibratedObservation]] = None,
) -> Dict[str, object]:
    """
    Evaluate the provenance invariants over a record set (plus, when
    supplied, its calibrated derived view). Returns a machine-readable
    table; the TEST SUITE asserts every invariant holds.

    Invariants evaluated here:
      A  no SIMULATED-sourced observation appears as MEASURED anywhere:
         raw provenance is SIMULATED -> derived label (if present) is
         SIMULATED or SIMULATED_CALIBRATED, never MEASURED*.
      B  UNAVAILABLE observations carry no value (unavailable stays
         unavailable; it can never become numeric zero or a distance).
      C  STALE observations carry no value (stale can never present as
         a fresh measurement).
      G  every calibrated label is exactly the legal mapping of its
         source provenance (identity keeps the vocabulary verbatim; a
         fitted transform maps MEASURED->MEASURED_CALIBRATED and
         SIMULATED->SIMULATED_CALIBRATED) - there is no free-form
         relabeling path.

    D (MEASURED raw -> MEASURED_CALIBRATED under a fitted transform),
    E (SIMULATED raw -> SIMULATED_CALIBRATED) and F (raw records
    byte-unchanged by calibration) are asserted directly against
    experiment outputs and record snapshots in the test suite.
    """
    violations: List[str] = []

    for r in records:
        key = f"{r.scene_id}/{r.frame_id}/{r.track_id}"
        if r.distance_provenance == "UNAVAILABLE":
            if r.predicted_distance_m is not None:
                violations.append(f"B: {key} carries a value")
        if r.distance_provenance == "STALE":
            if r.predicted_distance_m is not None:
                violations.append(f"C: {key} carries a value")

    sim_relabel: List[str] = []
    mapping_problems: List[str] = []
    if calibrated is not None:
        for obs in calibrated:
            key = f"{obs.scene_id}/{obs.frame_id}/{obs.track_id}"
            label = obs.calibrated_provenance
            src = usable_provenance_for_scoring(label)
            legal = {
                "MEASURED": ("MEASURED", CALIBRATED_PROVENANCE),
                "SIMULATED": ("SIMULATED", CALIBRATED_PROVENANCE_SIMULATED),
                "UNAVAILABLE": ("UNAVAILABLE",),
                "STALE": ("STALE",),
            }.get(src)
            if legal is None:
                mapping_problems.append(
                    f"G: {key} label {label!r} has no source mapping"
                )
            elif label not in legal:
                mapping_problems.append(
                    f"G: {key} label {label!r} is not derived from "
                    f"{src!r}"
                )
            if (obs.raw_provenance == "SIMULATED"
                    and label in ("MEASURED", CALIBRATED_PROVENANCE)):
                sim_relabel.append(f"A: {key} SIMULATED became {label!r}")

    return {
        "invariants": {
            "A_simulated_never_measured": not sim_relabel,
            "B_unavailable_stays_unavailable": not any(
                v.startswith("B:") for v in violations
            ),
            "C_stale_never_fresh": not any(
                v.startswith("C:") for v in violations
            ),
            "G_provenance_derived_not_relabeled": not mapping_problems,
        },
        "violations": violations + sim_relabel + mapping_problems,
    }


# --------------------------------------------------------------
# Dataset acceptance (Phase 12 §12)
# --------------------------------------------------------------

class DatasetAcceptanceError(ValueError):
    """A dataset is in an INVALID-AUDIT-STATE (structurally unusable)."""


@dataclass(frozen=True)
class DatasetAcceptance:
    """
    Verdict for a future PHYSICAL dataset intended for calibration.

    Two distinct failure classes (Phase 12 §12):

    - INVALID-AUDIT-STATE (raises DatasetAcceptanceError): structural
      preconditions of an auditable experiment are violated - split
      overlap, provenance outside the taxonomy, unfrozen protocol
      where freeze is required.
    - INVALID-FOR-FIT (reported, not raised): audit-present
      observations that must not enter fitting - stale/unavailable
      values, missing truth, invalid numerics. Visible via
      eligibility, never silently dropped.
    """

    accepted: bool
    invalid_for_fit: Tuple[Tuple[str, str, int, Optional[int], str], ...]
    summary: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "accepted": self.accepted,
            "n_invalid_for_fit": len(self.invalid_for_fit),
            "invalid_for_fit": [
                {"dataset": d, "scene_id": s, "frame_id": f,
                 "track_id": t, "reason": r}
                for d, s, f, t, r in self.invalid_for_fit
            ],
            "summary": self.summary,
        }


def accept_dataset(
    calibration_records: Sequence[MeasurementRecord],
    validation_records: Sequence[MeasurementRecord],
    protocol: Optional[ExperimentProtocol] = None,
    require_frozen: bool = True,
    now: Optional[float] = None,
    stale_after_s: float = 2.0,
    requested_targets_m: Optional[Sequence[float]] = None,
    coverage_tolerance_m: float = 0.05,
) -> DatasetAcceptance:
    """
    Accept (or loudly reject) a dataset pair for calibration work.

    Raises DatasetAcceptanceError (INVALID-AUDIT-STATE) when:
      - calibration and validation records OVERLAP (leakage);
      - any provenance is outside the six-value taxonomy;
      - require_frozen and the protocol is not frozen.

    Returns a DatasetAcceptance: accepted means "structurally sound
    AND fit-eligible records exist in both sets (and all requested
    experiment targets are covered, when requested)". It does NOT
    mean the data is physically accurate. Partial staleness is
    reported via invalid_for_fit; a set with NO eligible records is
    not accepted.
    """
    # ---- Structural gate 1: leakage -------------------------------
    try:
        assert_disjoint(calibration_records, validation_records)
    except ValueError as exc:
        raise DatasetAcceptanceError(
            f"INVALID-AUDIT-STATE: calibration/validation overlap - {exc}"
        ) from exc

    # ---- Structural gate 2: provenance taxonomy --------------------
    for role, records in (("calibration", calibration_records),
                          ("validation", validation_records)):
        for r in records:
            if r.distance_provenance not in ALL_PROVENANCES:
                raise DatasetAcceptanceError(
                    f"INVALID-AUDIT-STATE: {role} record "
                    f"{r.identity_key()} has unknown provenance "
                    f"{r.distance_provenance!r}"
                )

    # ---- Structural gate 3: freeze where required -------------------
    if protocol is not None and require_frozen and not protocol.frozen:
        raise DatasetAcceptanceError(
            "INVALID-AUDIT-STATE: protocol is not frozen; a physical "
            "calibration dataset requires a frozen capture protocol"
        )

    # ---- Fit-eligibility population (reported, not raised) ----------
    invalid_for_fit: List[Tuple[str, str, int, Optional[int], str]] = []
    eligible_by_role: Dict[str, List[MeasurementRecord]] = {}
    verdict_counts: Dict[str, Dict[str, int]] = {}

    for role, records in (("calibration", calibration_records),
                          ("validation", validation_records)):
        report = calibration_eligibility(
            records, now=now, stale_after_s=stale_after_s
        )
        eligible_by_role[role] = list(report.eligible)
        invalid_for_fit.extend(
            (role,) + entry for entry in report.excluded
        )
        counts: Dict[str, int] = {}
        for r in records:
            v = check_observation(r, now=now, stale_after_s=stale_after_s)
            counts[v.classification] = counts.get(v.classification, 0) + 1
        verdict_counts[role] = counts

    accepted = bool(eligible_by_role["calibration"]) and bool(
        eligible_by_role["validation"]
    )

    summary: Dict[str, object] = {
        "n_calibration": len(calibration_records),
        "n_validation": len(validation_records),
        "fit_eligible_calibration": len(eligible_by_role["calibration"]),
        "fit_eligible_validation": len(eligible_by_role["validation"]),
        "verdicts": verdict_counts,
        "protocol_frozen": None if protocol is None else protocol.frozen,
    }
    if requested_targets_m is not None:
        missing = _missing_targets(
            eligible_by_role["calibration"] + eligible_by_role["validation"],
            requested_targets_m, coverage_tolerance_m,
        )
        summary["requested_targets_m"] = sorted(set(requested_targets_m))
        summary["missing_targets_m"] = missing
        summary["coverage_note"] = (
            "EXPERIMENT TARGET coverage - analysis gate, not a safety "
            "threshold"
        )
        if missing:
            accepted = False

    return DatasetAcceptance(
        accepted=accepted,
        invalid_for_fit=tuple(invalid_for_fit),
        summary=summary,
    )


def _missing_targets(
    records: Sequence[MeasurementRecord],
    requested_targets_m: Sequence[float],
    tolerance_m: float,
) -> List[float]:
    return sorted(
        t for t in set(requested_targets_m)
        if not any(
            r.ground_truth_distance_m is not None
            and abs(r.ground_truth_distance_m - t) <= tolerance_m
            for r in records
        )
    )


# --------------------------------------------------------------
# Software readiness (Phase 12 §14/§15) - TWO DISTINCT GATES
# --------------------------------------------------------------

SOFTWARE_READINESS_CHECKS: Tuple[str, ...] = (
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
)


def _ready_record(
    provenance: str = "MEASURED",
    frame: int = 1,
    pred: float = 2.0,
    truth: Optional[float] = 2.0,
    scene: str = "sim_readiness",
) -> MeasurementRecord:
    """Tiny deterministic record for readiness self-checks."""
    return MeasurementRecord(
        schema_version=1,
        scene_id=scene,
        frame_id=frame,
        timestamp=100.0 + frame,
        track_id=1,
        label="person",
        confidence=0.9,
        bbox=(0, 0, 40, 120),
        region="CENTER",
        predicted_distance_m=pred,
        distance_provenance=provenance,
        distance_source="readiness-check",
        ground_truth_distance_m=truth,
    )


def software_calibration_readiness() -> Dict[str, object]:
    """
    SOFTWARE-ONLY readiness check (Phase 12 §14): is the capture /
    calibration pipeline ready to accept a future measured dataset?

    It verifies software capabilities by EXERCISING them with a tiny
    in-memory scenario (no files, no hardware, no network). It
    deliberately does NOT and CANNOT answer whether OAK-D hardware
    works: that is the separate, currently BLOCKED hardware gate
    (Phase 9 Checkpoint A). The result carries both gates explicitly
    so they can never be merged (Phase 12 §15).
    """
    checks: Dict[str, bool] = {}
    details: Dict[str, str] = {}

    def _run(name: str, fn) -> None:
        try:
            fn()
            checks[name] = True
        except Exception as exc:  # readiness reports, never crashes
            checks[name] = False
            details[name] = f"{type(exc).__name__}: {exc}"

    # ---- Provider contract + vocabulary -----------------------------
    def _provider_contract() -> None:
        from depth.provider import DepthMeasurement, DistanceProvenance
        m = DepthMeasurement(
            distance_m=1.0,
            provenance=DistanceProvenance.MEASURED,
            source="readiness-check",
        )
        assert m.usable and m.provenance_str == "MEASURED"

    _run("provider_contract_importable", _provider_contract)

    def _measured_supported() -> None:
        for prov in ("MEASURED", "SIMULATED", "UNAVAILABLE", "STALE"):
            assert prov in SOURCE_PROVENANCES
        for prov in CALIBRATED_PROVENANCES:
            assert prov not in SOURCE_PROVENANCES

    _run("measured_provenance_supported", _measured_supported)

    def _calibrated_defined() -> None:
        rec = _ready_record("MEASURED")
        frozen = IdentityCalibration().fit([rec])
        obs = apply_calibration([rec], frozen)[0]
        # Identity keeps the vocabulary verbatim; raw value preserved.
        assert obs.calibrated_provenance == "MEASURED"
        assert obs.raw_predicted_m == rec.predicted_distance_m

    _run("calibrated_provenances_defined", _calibrated_defined)

    # ---- Capture protocol lifecycle ----------------------------------
    def _capture_protocol() -> None:
        plan = TargetDistancePlan(targets_m=(1.0,), repetitions=1)
        protocol = ExperimentProtocol(
            plan=plan, scenes=("sim_readiness",), experiment_id="readiness"
        )
        capture = protocol.start_capture("sim_readiness", 1.0, 1)
        rec = _ready_record()
        capture.add_records([rec])
        capture.complete()
        protocol.verify_split()  # empty other side is trivially disjoint
        protocol.freeze()
        assert protocol.frozen

    _run("capture_protocol_available", _capture_protocol)

    def _truth_attachment() -> None:
        plan = TargetDistancePlan(targets_m=(1.0,), repetitions=1)
        protocol = ExperimentProtocol(
            plan=plan, scenes=("sim_readiness",), experiment_id="readiness"
        )
        capture = protocol.start_capture("sim_readiness", 1.0, 1)
        rec = _ready_record(truth=None)  # truth must arrive via attachment
        capture.add_records([rec])
        joined = capture.attach_ground_truth(
            {(rec.scene_id, rec.frame_id, rec.track_id): 1.97}
        )
        assert joined == 1
        derived = capture.derived_records()[0]
        assert derived.ground_truth_distance_m == 1.97
        assert capture.records[0].ground_truth_distance_m is None

    _run("ground_truth_attachment_available", _truth_attachment)

    def _split_enforcement() -> None:
        a = _ready_record()
        b = MeasurementRecord(
            schema_version=1, scene_id=a.scene_id, frame_id=a.frame_id,
            timestamp=a.timestamp, track_id=a.track_id, label=a.label,
            confidence=a.confidence, bbox=a.bbox, region=a.region,
            predicted_distance_m=2.5, distance_provenance="MEASURED",
            distance_source="x", ground_truth_distance_m=2.0,
        )
        try:
            assert_disjoint([a], [b])
        except ValueError:
            return
        raise AssertionError("overlap was not rejected")

    _run("split_enforcement_available", _split_enforcement)

    def _manifest_persistence() -> None:
        plan = TargetDistancePlan(targets_m=(1.0,), repetitions=1)
        protocol = ExperimentProtocol(
            plan=plan, scenes=("sim_readiness",), experiment_id="readiness"
        )
        capture = protocol.start_capture("sim_readiness", 1.0, 1)
        capture.add_records([_ready_record()])
        capture.complete()
        protocol.freeze()
        round_trip = json.loads(json.dumps(protocol.manifest()))
        rehydrated = ExperimentProtocol.from_manifest(round_trip)
        assert rehydrated.frozen

    _run("manifest_persistence_available", _manifest_persistence)

    def _dataset_freeze() -> None:
        from calibration.protocol import ProtocolFrozenError
        plan = TargetDistancePlan(targets_m=(1.0,), repetitions=1)
        protocol = ExperimentProtocol(
            plan=plan, scenes=("sim_readiness",), experiment_id="readiness"
        )
        protocol.freeze()
        try:
            protocol.start_capture("sim_readiness", 1.0, 1)
        except ProtocolFrozenError:
            return
        raise AssertionError("frozen protocol accepted a new capture")

    _run("dataset_freeze_available", _dataset_freeze)

    def _report_generation() -> None:
        from calibration.reporting import build_experiment_report
        # Distinct observations for the two splits (disjoint identities).
        calibration_rows = [
            _ready_record(frame=1, truth=1.0, pred=1.1),
            _ready_record(frame=2, truth=2.0, pred=2.2),
        ]
        validation_rows = [
            _ready_record(frame=11, truth=1.0, pred=1.1),
            _ready_record(frame=12, truth=2.0, pred=2.2),
        ]
        result = CalibrationExperiment(
            candidates=[IdentityCalibration()]
        ).run(calibration_rows, validation_rows)
        report = build_experiment_report(result)
        assert "validation_classification" in report

    _run("report_generation_available", _report_generation)

    def _raw_derived_separation() -> None:
        rec = _ready_record(frame=3, pred=2.0, truth=2.0)
        snapshot = rec.to_json()
        fit_rows = [
            _ready_record(frame=1, truth=1.0, pred=1.1),
            _ready_record(frame=2, truth=2.0, pred=2.2),
        ]
        frozen = AffineCalibration().fit(fit_rows)
        apply_calibration([rec], frozen)
        assert rec.to_json() == snapshot  # raw record untouched

    _run("raw_derived_separation_available", _raw_derived_separation)

    passed = all(
        checks.get(name, False) for name in SOFTWARE_READINESS_CHECKS
    )
    return {
        "schema": "sina-software-readiness/1",
        "software_calibration_readiness": "PASS" if passed else "FAIL",
        "checks": {name: checks.get(name, False)
                   for name in SOFTWARE_READINESS_CHECKS},
        "failed_check_details": details,
        # THE TWO GATES, EXPLICITLY SEPARATE (Phase 12 §15):
        "hardware_checkpoint": "BLOCKED",
        "hardware_note": (
            "Software readiness NEVER implies hardware validation. The "
            "OAK-D stereo checkpoint (Phase 9 Checkpoint A) remains "
            "BLOCKED and is the ONLY gate that can unlock real "
            "MEASURED data."
        ),
        "classification": {
            "software": "SOFTWARE-VALIDATED",
            "hardware": "NOT VALIDATED - Phase 9 Checkpoint A BLOCKED",
            "physical_distance_accuracy": "NOT VALIDATED",
            "safety": "NOT VALIDATED",
        },
    }
