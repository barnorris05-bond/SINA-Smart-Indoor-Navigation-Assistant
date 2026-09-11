# Changelog

All notable changes to the Smart Indoor Navigation Assistant (SINA) project will be documented in this file.

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
