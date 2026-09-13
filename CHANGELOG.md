# Changelog

All notable changes to the Smart Indoor Navigation Assistant (SINA) project will be documented in this file.

---

## - 2026-09-13 (Phase 8 — Temporal Navigation Stabilization)
### Added
- **TemporalNavigator** (`navigation/temporal_navigator.py`): stabilizing wrapper that composes (never replaces) DistanceNavigator — the candidate decision per frame is accepted, held, or confirmed before reaching audio. Answers "what should SINA do over time?"; does not recompute risk, re-track objects, or generate speech.
- **Total severity order** (no second hierarchy invented): `severity = max(action_floor, scene_max_risk)` where action floors are STOP=4, SLOW_DOWN/MOVE_*=2, CONTINUE=1 and scene risk reuses the Phase 7 `RiskLevel` ordering. Fixes a draft flaw where, with no risk data, a genuine lateral hazard (CONTINUE → MOVE_LEFT) could be suppressed forever.
- **Scene risk is a MAX, not a mean**: one CRITICAL object makes the scene CRITICAL — a low-risk object can never dilute a high-risk one.
- **Scene-CRITICAL forces STOP** (`CRITICAL_RISK_FORCE_STOP`, development/test): a scene whose max annotated risk is CRITICAL produces STOP even when the distance-band candidate was weaker (fast approach just outside the STOP band). This is the point where risk annotation starts to influence navigation; RiskEstimator remains the sole classification authority.
- **Escalation is immediate**: any severity increase is displayed the frame it appears; every STOP candidate is force-accepted (`CRITICAL_OVERRIDE`) — a genuine hazard is never delayed by smoothing, including a new obstacle arriving while already stopped.
- **De-escalation requires persistence** (`DEESCALATION_CONFIRM_FRAMES`=3, aligned with the risk layer's anti-flap window so navigation never relaxes faster than its risk input): a lower-severity candidate must appear on 3 consecutive frames; the previous action is re-emitted meanwhile with an auditable "holding … pending de-escalation confirmation (k/N)" reason. One noisy low frame can never erase a confirmed STOP; a hazard returning mid-confirmation resets the counter.
- **Equal-severity action changes require persistence** (`ACTION_CHANGE_CONFIRM_FRAMES`=2, development/test): covers MOVE_LEFT ↔ MOVE_RIGHT direction flips AND forward ↔ lateral changes — alternating noisy candidates never chatter direction, a persistent genuine change switches after 2 consecutive frames, and re-confirmation of the current action resets the streak (strictly-consecutive semantics).
- **Legacy fallback preserved**: with no usable distance the candidate comes from the spatial Navigator inside DistanceNavigator and is stabilized like any other candidate; risk annotation is optional (action floors alone still stabilize).
- **Data honesty**: decision logic only — never fabricates distances, never converts SIMULATED→MEASURED or STALE→current, never mutates detections; held decisions re-emit the current (action, trigger) signature so existing audio dedup/cooldown behavior is unchanged.
- **Test suite** (`tests/test_temporal_navigator.py`): 38 hardware-free tests — stable scenes, immediate escalation, immediate critical STOP, confirmed de-escalation (exact policy verified), risk-spike streak reset, STOP/SLOW_DOWN and CONTINUE/SLOW_DOWN flap suppression, directional jitter suppression + genuine direction change + streak reset, forward↔lateral gating, scene-CRITICAL force-STOP, monotonic escalation ladder, max-risk scene semantics, legacy fallback parity (temporal output equals legacy Navigator actions under unavailable depth), track disappearance (STOP held through a 1-frame detection loss, track id survives), single-frame distance loss, provenance non-promotion, audio-signature stability during holds, reset semantics, no cross-track leakage, config guards, and a full E2E hazard (2.8→1.0 m sustained approach: track #1 constant, SIMULATED provenance preserved, monotonic escalation to STOP).

### Changed
- `main.py`: `navigator = TemporalNavigator(DistanceNavigator())` — one instance for the application lifetime in both loops (camera and fallback). Legacy Navigator untouched inside the composition.
- `tests/conftest.py`: temporal suite added to the pytest allow-list.

### Verified
- 199/199 hardware-free pytest suite (161 pre-existing + 38 temporal).
- Temporal layer costs ~0.004 ms/frame including the candidate decision (≈0.006% of the 15 FPS budget); p95 ≈ 0.005 ms.
- `main.py` smoke test: clean no-device fallback, full annotated chain (tracker → fusion → motion → risk → temporal navigation → audio) runs, audio dedup intact, graceful shutdown.
- DEVELOPMENT/TEST temporal stabilization only: hysteresis values are not calibrated; demonstrates reduced decision jitter and immediate escalation under SIMULATED conditions. Real-world stabilization/safety validation requires MEASURED stereo and Phases 9–13.

---

## - 2026-09-13 (Phase 7 — Risk Estimation)
### Added
- **RiskEstimator** (`navigation/risk_estimator.py`): per-object risk classification — `LOW / MEDIUM / HIGH / CRITICAL` — answering "how dangerous is this?" as a layer deliberately separate from MotionEstimator ("what is it doing?"), DistanceNavigator ("what should SINA do?") and AudioManager ("what does the user hear?"). No navigation logic lives in the estimator; no risk logic in tracker/fusion/motion.
- **Transparent rule ladder** (auditable reasons on every assessment): CRITICAL = distance ≤ `CRITICAL_DISTANCE_M` or TTC ≤ `TTC_CRITICAL_S`; HIGH = TTC ≤ `TTC_WARNING_S`, or warning-band distance + (high-priority OR APPROACHING), or high-priority approaching within FAR; MEDIUM = warning-band distance or high-priority within FAR; LOW otherwise. NOT distance-only: motion and priority escalate; a stationary chair and an approaching person at the same distance classify differently.
- **Threshold alignment**: distance bands imported from `config/navigation_distance.py`, high-priority threshold from `config/navigation.py` — risk and navigation share one source of truth and cannot silently diverge. Risk-native tunables in `config/risk.py` (TTC bands, TTC minimum rate, anti-flap confirm frames), all labeled DEVELOPMENT/TEST.
- **Trustworthy TTC only**: `TTC ≈ distance / closing_rate` computed only when motion state is hysteresis-confirmed APPROACHING with rate ≥ `TTC_MIN_CLOSING_RATE_MPS` (0.1 m/s). STALE/UNAVAILABLE/UNKNOWN/receding/tiny-rate inputs yield `ttc_s=None` — never a meaningless TTC. TTC is documented as an ESTIMATE, not a guaranteed collision time.
- **Conservative unknown**: without usable distance (same usability rule as DistanceNavigator — shared helper `distance_usable`), high-priority obstacles rate MEDIUM (never silently LOW); low-priority rate LOW.
- **Confidence never lowers risk**: low-confidence detections only add an advisory reason; a shaky person detection at 0.9 m stays CRITICAL.
- **Anti-flap**: escalation is immediate; de-escalation requires `RISK_DROP_CONFIRM_FRAMES` (3) consecutive lower-level proposals (per track; one noisy frame cannot turn CRITICAL into LOW; a single higher proposal resets the pending drop). Untracked objects are assessed statelessly.
- **Provenance honesty**: assessment provenance is MEASURED only from genuinely measured distances (SIMULATED under mock mode); unusable inputs are UNAVAILABLE — mirroring the Phase B–D discipline.
- **Test suite** (`tests/test_risk_estimator.py`): 27 hardware-free tests — the full §9 scenario matrix (far/near/very-near × stationary/approaching/receding, stale, unavailable, unknown motion, low confidence), TTC gating and boundaries, anti-flap behavior (instant escalation, confirmed de-escalation, monotonic severity under sustained approach), threshold-alignment parity with navigation, and full-pipeline integration (DetectionManager → ObjectTracker → DistanceFusion → MotionEstimator → RiskEstimator) proving navigation decisions byte-identical with and without the risk layer.

### Changed
- `vision/object_detector.py`: `DetectedObject` gained `risk_level`, `risk_reasons`, `risk_ttc_s`, `risk_provenance` (additive, default None — backward compatible).
- `main.py`: `risk.update(detections)` called after motion, before navigation, in both loops — annotation only this phase (Phase 8 wires risk into temporal navigation). Navigation/audio semantics unchanged.
- `tests/conftest.py`: risk suite added to the pytest allow-list.

### Verified
- 161/161 hardware-free pytest suite (134 pre-existing + 27 risk). RiskEstimator costs ~0.025 ms/frame for 8 objects (0.04% of the 15 FPS budget).
- `main.py` smoke test: clean no-device fallback, risk annotation live in-loop, no behavioral change to navigation or audio.
- SIMULATION-VALIDATED only: risk behavior is demonstrated on simulated distances/motion. Real-world risk calibration depends on MEASURED stereo and calibrated thresholds (Phases 9–10); current levels are not safety-calibrated.

---

## - 2026-09-13 (Phase 6 — Motion / Approach Estimation)
### Added
- **MotionEstimator** (`vision/motion_estimator.py`): per-track temporal classification of distance behavior — `APPROACHING / STATIONARY / RECEDING / UNKNOWN` plus `closing_rate_mps` (positive = distance decreasing = approaching). Dedicated module: no motion logic in tracker, fusion, navigation or audio; no risk logic (Phase 7).
- **Distance history with validity gating** (`config/motion.py`): per-track samples (timestamp, distance, provenance); only MEASURED/SIMULATED distances with positive values enter history — `None`/UNAVAILABLE/STALE never become motion evidence, and valid streams are not poisoned by stale frames.
- **Two-half median rate filter**: the estimation window is split into first/second halves; rate = (median distance of first half − median of second half) / (median timestamp difference). Medians reject single-frame depth outliers deterministically; half-center pairing has no lag bias for linear motion.
- **Hysteresis bands**: switches INTO APPROACHING/STATIONARY/RECEDING only when the estimated rate is inside that band; dead-zone rates keep the previous state (no classification flicker from noisy rates).
- **Gap safety**: rates use only samples within `WINDOW_SPAN_S` (3 s). After a long detection/staleness gap the estimator honestly reports UNKNOWN until fresh evidence re-accumulates — pre-gap rates are never blended across a gap.
- **Provenance honesty for motion**: estimate provenance is MEASURED only when ALL contributing samples are MEASURED; any SIMULATED sample makes the estimate SIMULATED (simulated approach speed can never claim to be a real measurement). Rates are real m/s via injectable clock, not frame-count guesses.
- **Track hygiene**: history purged after `MAX_HISTORY_AGE_FRAMES` (15) consecutive absences; `reset()` clears all state.
- **Test suite** (`tests/test_motion_estimator.py`): 23 hardware-free tests — all four states, insufficient history/observation span, noise robustness (no oscillation), hysteresis dead zones, sustained-change reclassification, STALE/UNAVAILABLE rejection, valid-stream integrity, provenance honesty (MEASURED/SIMULATED/mixed), multi-track independence, gap handling, expiry purge, reset, additive-only field guarantee, and full-pipeline integration (DetectionManager → ObjectTracker → DistanceFusion → MotionEstimator → DistanceNavigator) proving navigation semantics unchanged.

### Changed
- `vision/object_detector.py`: `DetectedObject` gained `motion_state`, `closing_rate_mps`, `motion_provenance` (additive, default None — fully backward compatible).
- `main.py`: single `MotionEstimator` instance; `motion.update(detections)` called after fusion, before navigation, in both loops. Navigation/audio untouched.
- `tests/conftest.py`: motion suite added to the pytest allow-list.

### Verified
- 134/134 hardware-free pytest suite (111 pre-existing + 23 motion). MotionEstimator costs ~0.046 ms/frame for 8 tracks (0.07% of the 15 FPS budget).
- `main.py` smoke test: clean no-device fallback, motion live in-loop, audio gating unchanged.
- SIMULATION-VALIDATED only: all motion behavior is demonstrated on scripted/simulated distances. Real-world closing-rate accuracy depends on MEASURED stereo (Phase 9–10) and is NOT validated.

---

## - 2026-09-11 (Phase 5 — Object Tracking)
### Added
- **Deterministic Object Tracker** (`vision/object_tracker.py`): persistent, monotonically increasing `track_id` assignment across frames. Hardware-independent (no DepthAI/YOLO/OpenCV/network), operates purely on `DetectedObject` streams, detection-order independent.
- **Matching algorithm — greedy two-pass, one-to-one**: pass 1 matches label-compatible pairs by descending IoU (`MIN_IOU`), pass 2 by ascending centroid distance (`MAX_CENTROID_DISTANCE`); a track matches at most one detection per frame and vice versa. Chosen over ByteTrack/BoT-SORT for determinism, zero dependencies and simple testability at the current prototype stage (motion-model trackers remain a documented escalation path if identity swaps become a measured problem).
- **Track lifecycle**: NEW (birth) → ACTIVE (matched) → MISSING (temporarily undetected, retained up to `MAX_MISSED_FRAMES` ≈ 0.7 s at 15 FPS) → EXPIRED (removed; IDs never reused). `reset()` clears all tracks with IDs staying monotonic.
- **Tracking configuration** (`config/tracking.py`): `MAX_MISSED_FRAMES`, `MIN_IOU`, `MAX_CENTROID_DISTANCE` — DEVELOPMENT/TEST tuning, explicitly not real-world-calibrated.
- **Tracking test suite** (`tests/test_object_tracker.py`): 29 hardware-free tests — persistence, movement, temporary disappearance, expiration, multi-object same-class, detection-order swaps, label compatibility, one-to-one under duplicates, deterministic crossing behavior, reset, and full data-integrity guarantees (distance / `distance_provenance` / `distance_source` untouched; SIMULATED/STALE never promoted; no distance invented for UNAVAILABLE), plus DetectionManager/ObjectTracker/DistanceFusion/DistanceNavigator integration and an approaching-person E2E (track 1,1,1,1 with navigation escalating CONTINUE→SLOW_DOWN→STOP on unchanged dev thresholds).

### Changed
- `main.py`: single `ObjectTracker` instance integrated in both loops at `DetectionManager → ObjectTracker → DistanceFusion` position; RGB/YOLO/mock-depth/navigation/audio behavior otherwise untouched.
- `tests/conftest.py`: tracking suite added to the pytest allow-list.

### Verified
- 111/111 hardware-free pytest suite (82 pre-existing + 29 tracking); tracker costs ~0.46 ms/frame at 20 tracks × 20 detections (99%+ headroom at the 15 FPS budget).
- `main.py` smoke test: clean no-device fallback, tracker live, audio gating unchanged.
- NOT validated: real-world tracking accuracy (no re-identification; crossing objects may swap identities at the crossing instant — geometric limitation, documented in tests).

---

## - 2026-09-11 (Phases B–D — Mock-Depth Development Stack)
### Added
- **DepthProvider Abstraction** (`depth/provider.py`): `DepthProvider` ABC, `DepthMeasurement` value type and the `DistanceProvenance` enum (`MEASURED / SIMULATED / UNAVAILABLE / STALE`) — the single boundary all downstream code depends on; DepthAI never leaks past it.
- **Mock Depth Providers** (`depth/mock_provider.py`): `ScenarioDepthProvider` (deterministic (label, region) table) and `BboxSyntheticProvider` (pinhole synthetic) behind a `MockDepthProvider` facade. Every measurement carries provenance `SIMULATED` and a source tag — mock values can never masquerade as real sensor readings.
- **Real Stereo Adapter — HARDWARE-UNVALIDATED** (`depth/oak_provider.py`): `OakStereoDepthProvider` adapts the gated `CameraManager` stereo stream to the provider interface using the `depth/validity.py` robust statistics. Implemented so the Phase E swap is a config flip; it has NOT been validated on hardware (stereo blocked at USB 2.0, previously unstable pipeline never started in this mode).
- **Provider Factory** (`depth/provider_factory.py`): single mock↔stereo swap point driven by `config/mock_depth.USE_MOCK_DEPTH`.
- **Distance Fusion** (`vision/distance_fusion.py`): per-object measurement attachment, provenance stamping, staleness rejection (values older than `MEASUREMENT_STALE_S` are withheld, never silently reused), and UNAVAILABLE passthrough with legacy fallback.
- **Distance-Aware Navigation** (`navigation/distance_navigator.py`): `STOP` ≤ 1.2 m / `SLOW_DOWN` or move-away ≤ 2.5 m / `CONTINUE` ≥ 4 m policy that delegates byte-identically to the legacy `Navigator` whenever no usable distance exists. Thresholds (`config/navigation_distance.py`) are DEVELOPMENT/TEST values, explicitly uncalibrated.
- **Simulated-Value UI Honesty**: overlay renders mock distances as `1.20 m (MOCK)` (`vision/drawing.py`).
- **Stereo Hardware Gate** (`config/camera.py`, `camera/camera_manager.py`): with `ENABLE_STEREO_HARDWARE=False` (current default) the previously unstable stereo pipeline is never built or started; depth stays `DEPTH_DISABLED` and RGB-only operation is the norm. `tests/test_depth_stream.py` sets the flag for its own session as the Phase E hardware test.
- **DepthAI-Free Mock Path**: `config/camera.py` no longer imports DepthAI at module level (stereo enums via lazy accessors); `import depth` pulls no hardware libraries in mock mode.
- **Tests**: `tests/test_mock_depth_stack.py` (35 no-hardware tests: provenance, both mock strategies, staleness, fusion, distance navigation, audio escalation, stereo-swap simulation) and `tests/test_pipeline_mock_e2e.py` (5 end-to-end pipeline tests with a fake detector). `tests/conftest.py` restricts pytest collection to these hardware-free suites (the directory mixes in device-opening probe scripts).

### Changed
- `main.py`: `DistanceNavigator` + fusion wired into both loops (`fusion.update()` → `fuse()` → `decide()` → `audio.update()`); RGB/YOLO/audio paths untouched.
- `vision/object_detector.py`: `DetectedObject` gained `distance_provenance` / `distance_source` (backward-compatible defaults).
- `vision/detection_manager.py`: provenance fields preserved through enrichment.

### Verified
- 81/81 hardware-free pytest suite (mock stack 35, e2e 5, depth validity 19, audio 23); Sprint 4 regression 5/5; live `main.py` boot verified clean no-device fallback (synthetic loop, gated audio).
- Semantic development scenarios: person 1.2 m → STOP, chair 2.5 m → SLOW_DOWN, obstacle 0.7 m → STOP (any region), 4.0 m → CONTINUE, unavailable → legacy fallback, stale → rejected.
- NOT validated: real stereo depth, real-world distance accuracy, stereo robustness, final safety thresholds (software-development scenarios only).

---

## - 2026-09-11 (Phase 2 — Stereo Depth Integration)
### Added
- **Aligned Stereo Depth Stream (Phase 2)**: `CameraManager` now builds a stereo branch in the same DepthAI 3.x pipeline as RGB: mono pair (CAM_B/CAM_C) at 640×400@15 GRAY8 → `StereoDepth` (FAST_DENSITY, left-right check, confidence threshold 200 via `initialConfig`) → depth aligned to CAM_A, yielding depth frames pixel-aligned with the 640×400 RGB frame.
- **Depth Sub-Status Health Model**: `DEPTH_DISABLED / DEPTH_WARMING_UP / DEPTH_OK / DEPTH_STALE` with warmup timing, measured FPS, frame counts, and staleness detection (>2s without frames); depth init failures degrade gracefully to RGB-only (logged, never crash).
- **Non-Blocking Depth Acquisition**: `DepthCamera.get_latest_frame()` drains the queue to the newest frame (no stale depth), returns None when empty, normalizes (1,H,W)→(H,W); `get_depth_frame()` never raises on device hiccups.
- **Robust Depth Measurement Module** (`depth/validity.py`): invalid-pixel masking (0 + configurable 200mm–20m range), central-ROI edge stripping, median-based region measurement, min-sample/min-valid-ratio rejection (“unknown is not safe”), frame-level validation reports.
- **Depth Validity Unit Tests** (`tests/test_depth_validity.py`): 19 no-hardware tests covering masking, ROI, robustness to outliers, rejection thresholds.
- **Phase 2 Hardware Test** (`tests/test_depth_stream.py`): production-path RGB+depth streaming test measuring warmup, FPS, validity, center-distance stability; exit codes 0=PASS / 1=FAIL / 2=NO DEVICE; `--preview` flag for visual check.

### Changed
- `config/camera.py`: mono reconfigured 1280×720@30 → 640×400@15 (RGB-aligned depth, unified 15 FPS, ~4× less USB bandwidth after prior X_LINK instability at high profiles; 720p documented as fallback); added stereo confidence threshold, depth validity range, ROI/valid-ratio thresholds, warmup constants.

### Verified
- 19/19 depth validity tests; 23/23 audio tests (regression); 5/5 Sprint 4 scenarios (regression); `main.py` boots and degrades gracefully with no device present.
- Hardware depth validation pending OAK-D reconnection (`tests/test_depth_stream.py` exits NO DEVICE until then).

---

## - 2026-09-11
### Added
- **Phase 1 — Audio Guidance (restored & verified)**: Rebuilt the missing Sprint 5 audio subsystem as `audio/` (`Speaker`, `Announcer`, `MessageBuilder`, `AudioManager`) after confirming it was never committed to the repository.
- **Asynchronous TTS Worker**: `audio/speaker.py` owns the pyttsx3 (SAPI5) engine on a dedicated daemon thread with a bounded queue; `speak()` never blocks the camera loop and speech failures cannot crash the vision pipeline.
- **Announcement Gating Policy**: `audio/announcer.py` implements state-signature deduplication (`ACTION|label|region|distance-bucket`), per-action cooldowns (STOP 1s, lateral 3s, CONTINUE 5s), priority-based safety escalation, and 0.5 m distance-closing escalation.
- **Anti-Jitter Distance Handling**: 0.5 m distance rounding plus near/mid/far bucketing so depth noise never produces speech chatter; singular/plural meter grammar handled.
- **Context-Aware Messages**: `NavigationDecision.trigger` (additive field) identifies the obstacle that caused a decision, enabling "Person at 1.5 meters ahead. Please stop." vs "Obstacle on your right. Move left." wording.
- **Audio Unit Tests**: `tests/test_audio_logic.py` — 23 deterministic tests with an injectable clock (no sleeping, no engine) covering dedup, cooldowns, escalation, suppression, and queue contracts.
- **Live TTS Smoke Test**: `tests/smoke_tts.py` verifies non-blocking updates (<1 ms), audible playback, exact gating over a clock-driven stream, and clean shutdown on real hardware.
- **Test Packaging**: `tests/__init__.py` added so logic tests import shared fixtures and are discoverable by pytest.

### Changed
- `main.py`: minimal integration — `AudioManager` created once, `audio.update(decision)` called per frame, `audio.shutdown()` in both `finally` paths. Camera/vision/navigation code untouched.
- `requirements.txt`: populated with actual runtime dependencies (pyttsx3 added).

### Fixed
- pyttsx3 `runAndWait()` driving pattern corrected after live testing exposed `"run loop not started"` engine errors; speech now plays reliably on SAPI5.

---

## - 2026-07-10
### Added
- **Asynchronous Audio Infrastructure**: Implemented a multi-threaded `Speaker` class utilizing a background `ThreadPoolExecutor` to offload heavy Text-to-Speech (`pyttsx3`) calls, maintaining a steady 30 FPS camera loop.
- **Context-Aware Event Signatures**: Introduced the `NavigationState` token structure tracking combinations of actions, labels, regions, and distance buckets to instantly detect spatial scene shifts.
- **Priority-Based Overrides**: Created a strict threat-level hierarchy (`PRIORITY_MAP`) allowing safety-critical alerts (e.g., `STOP`) to instantly bypass or interrupt active voice cooldowns.
- **Telemetry Anti-Jitter**: Added 0.5-meter distance rounding and range bucketing (`near`, `mid`, `far`) to silence repetitive speech stutter caused by minor visual sensor noise.
- **Unified Audio Orchestration**: Built the `AudioManager` to coordinate phrase lookups and state management through a single interface, isolating `main.py` from underlying speech engine logic.
- **Automated Verification Testing**: Deployed automated validation scripts (`test_audio.py`, `test_state_audio.py`, and `test_production_audio.py`) to systematically verify time-gating, state switches, and priority overrides.

### Changed
- Refactored `main.py` to cleanly initialize, handle, and close down the `AudioManager` resources within both the live camera feed pipeline and the synthetic fallback loop.
