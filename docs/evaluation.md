# Offline Measurement & Evaluation Infrastructure

Status: **SOFTWARE-VALIDATED / SIMULATION-VALIDATED only.**
Real OAK-D stereo depth remains hardware-BLOCKED (Phase 9 Checkpoint A not
performed). Nothing in this document claims measured distance accuracy or
safety.

## Purpose

When real stereo depth is eventually validated, this infrastructure answers
*"how accurate is the measured distance?"* — not just *"does the stereo
stream run?"* It is built now, hardware-free, so that the moment real
OAK-D data exists it can flow straight into evaluation without redesign:

```
REAL OAK-D  (Phase 9+, BLOCKED)            SIMULATED (today)
      |                                          |
      v                                          v
MeasurementRecord (JSONL)  <--------- offline replay / injection
      |
      v
evaluate() -> EvaluationSummary -> report (JSON)
```

## Measurement record schema (version 1)

One JSONL line per (frame, object) observation. Fields:

| field | type | meaning |
|---|---|---|
| `schema_version` | int | 1 |
| `scene_id` | str | logical capture session / scenario id |
| `frame_id` | int | monotonically increasing frame number |
| `timestamp` | float | seconds (synthetic in replay, real clock in capture) |
| `track_id` | int \| null | persistent tracker id; null = tracker not run |
| `label` | str | object class |
| `confidence` | float | detector confidence |
| `bbox` | [x1,y1,x2,y2] | RGB-frame bounding box |
| `region` | str | LEFT / CENTER / RIGHT |
| `predicted_distance_m` | float \| null | SINA's distance estimate; null = none |
| `distance_provenance` | str | MEASURED / SIMULATED / UNAVAILABLE / STALE |
| `distance_source` | str \| null | provider id (e.g. `oak_stereo`, `mock_scenario`) |
| `ground_truth_distance_m` | float \| null | manual/measured truth; **null = not measured** |

Rules:

- Ground truth is a **separate field** and is never fabricated: a missing
  truth stays `null` (never 0, never a guess).
- Provenance strings reuse the project vocabulary from
  `depth/provider.DistanceProvenance` verbatim. Deserialization rejects
  unknown provenance, malformed bboxes, and records from newer schemas.
- CSV form (`csv_header()` / `to_csv_row()`) is provided for spreadsheet
  inspection; JSONL is the primary, append-friendly format.

## Provenance semantics

The evaluation layer never reinterprets provenance — it only filters and
reports it:

- `MEASURED` — value originated from a real stereo measurement
  (stamped by `OakStereoDepthProvider`; hardware-unvalidated today).
- `SIMULATED` — value came from a mock/scripted provider.
- `UNAVAILABLE` — no distance this frame.
- `STALE` — stream was quiet; old values are never reused as current.

`evaluate(..., provenances=("MEASURED",))` restricts metrics to one
provenance. Without filtering, the summary lists every contributing
provenance and `validation_class()` labels what the data can honestly
claim:

| contributing provenances | validation_class |
|---|---|
| `{"SIMULATED"}` | SIMULATION-VALIDATED |
| `{"MEASURED"}` | MEASUREMENT-VALIDATED (per recorded provenance) |
| mixed | MIXED PROVENANCES — split before comparing |
| none usable | NO USABLE MEASUREMENTS |

"MEASUREMENT-VALIDATED" refers to provenance recorded by the capture
pipeline; it is only as honest as the capture that stamped it.

## Metric definitions

Given predicted `p` and ground truth `gt` (both > 0, meters):

```
absolute_error = |p - gt|
relative_error = |p - gt| / gt
MAE            = mean(absolute_error)
RMSE           = sqrt(mean(absolute_error^2))
median_abs_err = median(absolute_error)
max_abs_err    = max(absolute_error)
median_rel_err = median(relative_error)
p95_abs_err    = 95th percentile of absolute_error (linear interpolation)
invalid_rate   = n_invalid / n_total
```

## Invalid-measurement policy

A record is **valid** for accuracy metrics iff ALL of:

1. `predicted_distance_m` is not null
2. `distance_provenance` in {`MEASURED`, `SIMULATED`}
3. `predicted_distance_m` > 0
4. `ground_truth_distance_m` is not null
5. `ground_truth_distance_m` > 0

Everything else is **invalid**: excluded from accuracy metrics and counted
in `invalid_rate`. UNAVAILABLE/STALE records and missing ground truth
never silently become zero errors.

## Calibration vs validation separation

Observation identity is `(scene_id, frame_id, track_id)`.

- `check_overlap(calibration, validation)` reports identical triples and
  shared scene ids.
- `assert_disjoint(calibration, validation)` **raises** when the same
  observation appears in both sets.

Never tune distance calibration on data used to validate it — the
framework makes that mistake an error, not a habit. Recommended Phase 10
split: separate capture sessions (or, minimally, disjoint frame ranges of
one session with different scenes) for calibration vs validation.

## Offline replay

`ReplayRunner(ReplayConfig(scene_id=...)).run(frames)` drives the REAL
decision pipeline — DetectionManager → ObjectTracker → DistanceFusion →
MotionEstimator → RiskEstimator → TemporalNavigator — over scripted
frames. Each `ScriptedFrame` lists detections `(label, bbox, confidence)`
and a per-label distance script for `ScriptedDepthProvider`:

| script spec | produced DepthMeasurement |
|---|---|
| `None` | UNAVAILABLE |
| `float v` | SIMULATED `v` |
| `("MEASURED", v)` | MEASURED `v` (synthetic value, real-provider contract — provenance-handling tests only) |
| `("STALE", v)` | STALE, `distance_m=None` (stream-quiet form) |
| `("OLD", v)` | SIMULATED `v` with an old timestamp (fusion demotes to STALE) |

Replay is fully deterministic: a synthetic clock advances exactly
1/frame_rate s per frame, components are constructed fresh per run, and
the same frames always produce identical records, events and decisions.
Provenance in the output records is copied verbatim from the real
`DistanceFusion` stamping — the replay never re-derives or edits it.

## Failure-injection coverage

Scenarios proven by `tests/test_evaluation_infra.py` through the real
pipeline: unavailable depth → spatial fallback; STALE withheld (never
reused); old-timestamp demoted to STALE by fusion; sudden near jump →
immediate STOP; SIMULATED→MEASURED transition stamps MEASURED;
MEASURED→UNAVAILABLE degrades without crash; disappearing/returning
tracks keep their id; approaching object escalates monotonically to STOP;
receding object eventually de-escalates; a low-risk object never dilutes
a critical scene.

## Offline analysis & reporting layer

On top of recording, `evaluation/analysis.py`,
`evaluation/navigation_analysis.py` and `evaluation/reporting.py`
implement the full analysis workflow:

```
recorded JSONL/CSV -> load -> validate -> filter -> evaluate -> summarize -> report
```

### Workflow

```python
from evaluation.analysis import load_jsonl_records, filter_provenance
from evaluation.reporting import analyze_dataset, build_report, render_markdown

records = load_jsonl_records("capture.jsonl")          # strict: malformed line -> DatasetError
records = filter_provenance(records, "MEASURED")       # optional pre-filter
analysis = analyze_dataset(records, events=None)        # structured analysis
report = build_report(analysis, dataset_role="VALIDATION")
print(render_markdown(report))
```

Or from the shell (no dependencies added):

```
python -m evaluation analyze capture.jsonl \
    [--events events.jsonl] \
    [--role CALIBRATION|VALIDATION|UNSPECIFIED] \
    [--name my_report] \
    [--json report.json] [--markdown report.md] [--csv summary.csv]
```

The CLI validates input, prints a concise deterministic summary, and
exits 2 with a one-line actionable error (never a traceback) on
missing/malformed/empty input.

### Validation triage

`validate_records()` triages every record into accuracy-valid or an
explicit invalid bucket — `missing_ground_truth`,
`non_positive_ground_truth`, `unavailable_prediction`,
`stale_prediction`, `non_positive_prediction`,
`non_usable_provenance_with_value`, `unknown_provenance` — counted in
the report's `validation_triage` and never silently zeroed.

### Report schema (JSON, deterministic, no wall-clock fields)

| section | content |
|---|---|
| `metadata` | name, source, `dataset_role`, schema version, limitations |
| `dataset_summary` | totals, valid/invalid predictions, invalid rate, provenance counts, scenes, tracks, labels, frame/timestamp ranges |
| `validation_triage` | invalid counts by reason |
| `provenance_summary` | per-provenance count + metrics; absent provenances shown as absent (e.g. `MEASURED: not available`) |
| `overall_metrics` | MAE / RMSE / median / max / p95 / relative / invalid rate + `validation_class` |
| `per_label_metrics` | sample/valid/invalid, MAE, RMSE, p95, note for insufficient data |
| `per_scene_metrics` | per-scene counts, provenance, metrics, labels |
| `distance_bins` | accuracy by ground-truth range (0–1, 1–2, …, >5 m, `no_ground_truth`) |
| `error_distribution` | absolute/relative error buckets, counts by distance range |
| `navigation_summary` | action counts, transitions, changes, stable runs, STOP frames, escalations (descriptive) |
| `risk_summary` | LOW/MEDIUM/HIGH/CRITICAL counts, per-object transitions, TTC min/median (descriptive) |
| `temporal_summary` | changes/frame, longest stable run, STOP runs, direction flips (descriptive) |

### Navigation / risk / temporal analysis is DESCRIPTIVE

These statistics quantify recorded behavior. They do **not** claim
navigation effectiveness or safety. `compare_datasets(baseline,
treatment)` produces an explicitly labeled *DESCRIPTIVE COMPARISON*
(equal scene scripts are the caller's responsibility); it never emits
"improvement" or "proven" language — a causal claim needs a controlled
comparison that this tooling does not perform.

### Performance (measured, synthetic 10k-record dataset)

load ≈ 120 ms · analyze ≈ 80 ms · report build < 1 ms · markdown < 1 ms
· JSON+CSV+MD writes ≈ 4 ms. No optimization needed.

### Limitations of this tooling

- **This tooling does not validate safety.**
- **This tooling does not validate real-world distance accuracy by
  itself** — accuracy conclusions require MEASURED data with real
  ground truth (Phase 10).
- Metrics over SIMULATED data are SIMULATION-VALIDATED only; the
  report states this in `limitations` and in every `validation_class`.
- Navigation/risk/temporal summaries require an event trace; without
  one those sections are `None`.

## Future OAK-D workflow (gated — do not skip steps)

1. **Phase 9 Checkpoint A** — `python tests\test_oak_stereo_diagnostic.py`
   must exit 0 (hardware functionality) before any capture.
2. Capture sessions via `MeasurementRecorder` (records + event trace),
   one `scene_id` per session; write JSONL.
3. Measure ground truth manually per `(scene_id, frame_id, track_id)`
   (tape measure / laser meter); join with `attach_truth()`.
4. Split: calibration set vs **independent** validation set;
   `assert_disjoint` must pass.
5. `evaluate()` per provenance; only `MEASURED` rows count toward
   distance-accuracy conclusions.
6. If accuracy is poor, adjust **depth** calibration in Phase 10 — never
   navigation thresholds to compensate.

Until step 1 passes on hardware, all evaluation results are
SIMULATION-VALIDATED at best.

## Phase 10 preparation: calibration experiment framework

**Status: SOFTWARE-VALIDATED (unit tests) / SIMULATION-VALIDATED
(synthetic fixtures) only.** Real OAK-D measurement calibration remains
blocked pending Phase 9 hardware validation. Nothing here claims
distance accuracy, calibration generalization, or safety.

The `calibration/` package implements the gated experiment workflow
that will consume REAL MEASURED observations when they exist:

```
KNOWN PHYSICAL DISTANCE (operator ground truth)
    -> REAL OAK-D MEASUREMENT (MEASURED; Phase 9+, BLOCKED)
    -> MeasurementRecord (reused verbatim from evaluation/)
    -> CalibrationExperiment.run(calibration, validation)
           assert_disjoint FIRST (overlap fails loudly, never removed)
           -> raw baseline metrics (ALWAYS reported, §10)
           -> candidate fit + selection on CALIBRATION data ONLY (§19)
           -> FROZEN model (validation stage cannot refit, §20)
           -> calibrated validation metrics, raw reported alongside
    -> deterministic experiment report (JSON / Markdown)
```

### Components

| module | purpose |
|---|---|
| `calibration/session.py` | `ExperimentSession` — opt-in metadata (device, DepthAI version, resolutions, lighting, operator notes). Unknown values stay `None`; unknown fields rejected on load. |
| `calibration/plan.py` | `TargetDistancePlan` — deterministic capture schedule (targets × repetitions). Targets are **EXPERIMENT TARGETS**, not navigation/safety thresholds (AST-enforced: the package imports nothing from `navigation`). |
| `calibration/coverage.py` | distance/scene/label/track coverage; requested targets kept **separate** from observed ground truth; availability (invalid rate) by distance/label/scene/provenance. |
| `calibration/diagnostics.py` | repeatability (per-target spread; below `min_samples` groups flagged `insufficient`, no statistics); descriptive outlier report; `filter_outliers()` — auditable only (rule + counts + before/after MAE; raw data never mutated). |
| `calibration/models.py` | `CalibrationModel` interface; `IdentityCalibration` (the baseline); `AffineCalibration` (closed-form least squares). `fit()` returns a `FrozenCalibrationModel` with **no refit path**. Fitted transforms map provenance to explicit `MEASURED_CALIBRATED` / `SIMULATED_CALIBRATED` labels — a corrected value can never masquerade as a raw sensor measurement. Candidate selection: lowest MAE on calibration data; tie → fewer parameters; tie → first registered. Validation data is structurally absent from selection. |
| `calibration/experiment.py` | `CalibrationExperiment.run()` — the gated pipeline above; `CalibratedObservation` keeps raw **and** calibrated values side by side (§8); inputs never mutated. |
| `calibration/reporting.py` | deterministic JSON report (16 sections) + Markdown renderer + validation classification. |

### Future protocol (gated — do not skip steps)

1. Phase 9 Checkpoint A passes on hardware (still BLOCKED).
2. For each `TargetDistancePlan` target: place target at known
   distance, capture ≥ `repetitions` observations per scene
   (`MeasurementRecorder`), record an `ExperimentSession`.
3. Measure ground truth manually; `attach_truth()` by identity key.
4. Split observations into CALIBRATION and **independent** VALIDATION
   sets (different scenes/frames — never the same observations).
5. `CalibrationExperiment(candidates=[...]).run(calib, val)`;
   compare frozen-model validation metrics against the raw baseline.
6. Only `MEASURED`-derived rows support real accuracy conclusions;
   synthetic fixtures remain SIMULATION-VALIDATED forever.

With no candidates supplied, the experiment is baseline-only — the
uncalibrated numbers are always on the record first (§10).

### Performance (measured, synthetic 9,984-record dataset)

fit + selection + both evaluations ≈ 92 ms · report ≈ 36 ms ·
repeatability ≈ 4 ms · availability ≈ 10 ms · outlier filter ≈ 5 ms.
No optimization needed.

## Phase 10 data-collection protocol: controlled capture + manifests

**Status: SOFTWARE-VALIDATED (unit tests) / SIMULATION-VALIDATED
(synthetic fixtures) only.** **REAL CAPTURE REQUIRES SUCCESSFUL PHASE 9
HARDWARE VALIDATION** (still BLOCKED). Nothing here performs or claims
physical measurement, calibration, or safety.

The `calibration.capture` / `calibration.protocol` modules implement
the operator-facing experiment lifecycle over the existing framework
(no duplicated data models — `MeasurementRecord`, `ExperimentSession`,
`TargetDistancePlan` are reused verbatim):

```
ExperimentProtocol(plan, scenes, experiment_id)
    -> start_capture(scene, target, rep)   slot PLANNED -> ACTIVE
    -> capture.add_records(...)            RAW records (never mutated)
    -> capture.attach_ground_truth(...)    audited (§11), operator-supplied
    -> capture.complete()/abort()          deterministic state machine
    -> assign(capture, CALIBRATION|VALIDATION)   immutable split
    -> verify_split()                      assert_disjoint, fails loudly
    -> freeze()                            DATASET FROZEN (§17)
    -> completion() / manifest() / to_json()
```

### Capture lifecycle (deterministic state machine)

`PLANNED -> ACTIVE -> {COMPLETED | INCOMPLETE | ABORTED}`;
`PLANNED -> ABORTED`. Terminal states accept nothing; invalid
transitions raise `CaptureStateError`; adding records or truth outside
ACTIVE raises `CaptureNotActiveError`. An interrupted capture is
never treated as completed.

- **Slots vs attempts**: slots are virtual (scene × target ×
  repetition from the plan); a slot with no attempt is PLANNED.
  `start_capture` on an ABORTED slot creates attempt #2 — the aborted
  attempt is preserved untouched and never merged into the resume
  (only the LATEST terminal attempt feeds a dataset).
- **Completeness rule (documented)**: ≥ `min_observations` records,
  each with known identity, project-vocabulary provenance, and finite
  timestamp — otherwise the capture becomes INCOMPLETE (kept and
  visible, never discarded). Ground truth is deliberately NOT required
  for capture completion; its absence surfaces in the completion
  report and as `ground_truth=None` (never 0).
- **Repetitions** are first-class slots — never overwritten,
  individually recoverable.
- **Target vs ground truth**: the slot's `target_distance_m` is the
  plan value; the operator's observed reference lives only on records
  (`ground_truth_distance_m`) and audit entries. Never merged.
- **Ground-truth audit**: every attachment records observation
  identity, value, source/operator note, and — only if the operator
  explicitly supplies it — an uncertainty. Never invented.

### Dataset handling

- **Assignment** is per-slot, explicit, and immutable (re-assignment
  to the other role is rejected); only terminal captures can be
  assigned. `dataset_records(role)` returns the DERIVED view (truth
  applied) of latest terminal attempts; raw records are never mutated.
- **`verify_split()`** runs `assert_disjoint` across the two roles —
  shared observation identities fail loudly (§13).
- **`freeze()`** makes captures, assignments, and truth immutable from
  the experiment layer (`ProtocolFrozenError` on any mutation).
  Manifests rehydrated via `from_manifest()` are always FROZEN audit
  copies (§17/§19).
- **Completion report** counts planned/completed/incomplete/aborted
  slots, missing ground truth, missing observations (exact record
  count), resumed slots, and assignment coverage — `fully_complete`
  is `False` while anything is missing (§15).
- **Manifest** (`sina-capture-manifest/1`): deterministic JSON with
  plan, scenes, planned slots, per-capture lifecycle + truth audit,
  and the assignment table (§14/§19). `software_version` and all
  environment fields stay `None` unless operator-supplied.

### Performance (measured, synthetic 160-slot experiment)

full capture run ≈ 2.8 ms · verify_split ≈ 2.3 ms · completion ≈ 3.5
ms · manifest ≈ 1.7 ms · to_json ≈ 5.9 ms (189 KB) · from_manifest
< 1 ms. No optimization needed.
