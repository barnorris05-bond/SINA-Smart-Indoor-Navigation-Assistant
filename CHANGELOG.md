# Changelog

All notable changes to the Smart Indoor Navigation Assistant (SINA) project will be documented in this file.

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
