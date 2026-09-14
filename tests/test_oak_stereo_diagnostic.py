"""
tests/test_oak_stereo_diagnostic.py

PHASE 9 — CHECKPOINT A: ISOLATED STEREO HARDWARE DIAGNOSTIC.

EXPLICIT-RUN ONLY: this file is NOT in tests/conftest.py COLLECTABLE,
so a plain `pytest` never collects or executes it. It never runs
unless invoked directly:

    ./.venv/Scripts/python.exe tests/test_oak_stereo_diagnostic.py
    ./.venv/Scripts/python.exe tests/test_oak_stereo_diagnostic.py --seconds 60

Scope (master roadmap §3/§5): HOST -> OAK-D -> mono cameras ->
StereoDepth -> depth frames -> statistics -> clean shutdown.

    NO YOLO. No navigation. No audio. No risk. No temporal layer.

Isolates hardware functionality from the rest of SINA. Reuses the
production CameraManager (no duplicated DepthAI logic, master §6);
the development config gate ENABLE_STEREO_HARDWARE is enabled HERE,
in this process only, never in the repository default.

Exit codes:
  0 = CHECKPOINT A PASSED (hardware functionality validated)
  1 = FAIL (device present but stereo did not meet criteria)
  2 = NO DEVICE (OAK-D not connected — HARDWARE BLOCKED, not a code
      failure; never retried in a loop)
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import depthai as dai

# Phase 9 hardware diagnostic: exercise the real stereo branch in THIS
# process only. Set BEFORE importing CameraManager (repository default
# stays ENABLE_STEREO_HARDWARE=False for mock-depth development).
import config.camera as _camcfg
_camcfg.ENABLE_STEREO_HARDWARE = True

from camera.camera_manager import CameraManager
from depth.validity import validate_depth_frame, measure_region

# Central 100x100 of the 640x400 RGB frame (reference region).
CENTER_BBOX = (270, 150, 370, 250)

MIN_DEPTH_FPS = 8.0          # dev/test gate: below ~half target -> FAIL
MIN_VALID_RATIO = 0.05       # dev/test gate: depth effectively empty below this
MIN_CONSECUTIVE_FRAMES = 10  # dev/test gate: continuity evidence


def print_device_info() -> None:
    """§5.2 — device information from a short-lived Device handle."""
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        return
    info = devices[0]
    print("[device] id       :", info.deviceId)
    try:
        with dai.Device(info) as dev:
            print("[device] platform :", dev.getPlatformAsString())
            print("[device] product  :", dev.getProductName())
            print("[device] usbSpeed :", dev.getUsbSpeed())
            for cam in dev.getConnectedCameraFeatures():
                print(
                    f"[device] {cam.socket}: {cam.sensorName} "
                    f"{cam.width}x{cam.height}"
                )
            pairs = [(p.left.name, p.right.name)
                     for p in dev.getAvailableStereoPairs()]
            print("[device] stereoPairs:", pairs or "NONE")
    except Exception as error:
        # Non-fatal: CameraManager will surface real errors at start().
        print(f"[device] info query failed: {error}")


def main() -> int:
    args = sys.argv[1:]
    measure_seconds = 10.0
    if "--seconds" in args:
        measure_seconds = float(args[args.index("--seconds") + 1])
    print(f"[phase9] ISOLATED STEREO DIAGNOSTIC ({measure_seconds:.0f}s window)")
    print("[phase9] scope: stereo hardware only — no YOLO/nav/audio/risk")

    # ---- §5.1: connect to exactly one OAK-D (single attempt) ----------
    try:
        devices = dai.Device.getAllAvailableDevices()
    except Exception as error:
        print(f"[preflight] device query failed: {error}")
        return 2
    if not devices:
        print("[preflight] NO OAK-D DEVICE FOUND.")
        print("           HARDWARE BLOCKED — NO VALIDATION PERFORMED.")
        print("           Connect the OAK-D and re-run this diagnostic.")
        return 2
    print(f"[preflight] {len(devices)} device(s) found; using the first.")
    print_device_info()

    # ---- §5.3: minimal stereo pipeline via production CameraManager ---
    camera = CameraManager()
    try:
        camera.start()
    except Exception as error:
        print(f"[FAIL] camera start failed: {error}")
        print("       (device was enumerable but pipeline boot failed —")
        print("        record the exact error; do NOT retry in a loop)")
        return 1

    depth_count = 0
    rgb_count = 0
    consecutive = 0
    max_consecutive = 0
    longest_gap_ticks = 0
    gap_ticks = 0
    valid_ratios = []
    frame_mins = []
    frame_medians = []
    frame_maxs = []
    center_medians_mm = []
    first_depth_at = None
    exceptions = 0
    shutdown_ok = False
    t_start = time.time()
    t_end = t_start + measure_seconds + 5.0  # warmup grace

    print(f"[test] streaming ~{measure_seconds:.0f}s (+warmup grace)...")

    try:
        while time.time() < t_end:
            loop_t0 = time.perf_counter()
            try:
                rgb = camera.get_rgb_frame()
                if rgb is not None:
                    rgb_count += 1
                depth = camera.get_depth_frame()
            except Exception as error:
                exceptions += 1
                print(f"[test] acquisition exception #{exceptions}: {error}")
                if exceptions >= 10:
                    print("[test] aborting: repeated acquisition exceptions")
                    break
                depth = None

            if depth is not None:
                depth_count += 1
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
                gap_ticks = 0
                if first_depth_at is None:
                    first_depth_at = time.time()
                    print(
                        f"[test] first depth frame after "
                        f"{first_depth_at - t_start:.1f}s | "
                        f"shape={depth.shape} dtype={depth.dtype}"
                    )
                report = validate_depth_frame(depth)
                valid_ratios.append(report["valid_ratio"])
                valid_px = depth[depth > 0]
                if valid_px.size:
                    frame_mins.append(int(valid_px.min()))
                    frame_medians.append(float(np.median(valid_px)))
                    frame_maxs.append(int(valid_px.max()))
                stats = measure_region(depth, CENTER_BBOX)
                if stats.valid:
                    center_medians_mm.append(stats.distance_mm)
            else:
                consecutive = 0
                gap_ticks += 1
                longest_gap_ticks = max(longest_gap_ticks, gap_ticks)

            elapsed = time.perf_counter() - loop_t0
            time.sleep(max(0.0, 1.0 / 30.0 - elapsed))

        elapsed_s = time.time() - t_start

        # ---- §5.12: clean shutdown ------------------------------------
        camera.stop()
        shutdown_ok = True
    except Exception as error:
        print(f"[FAIL] unhandled diagnostic error: {error}")
        elapsed_s = time.time() - t_start
        try:
            camera.stop()
            shutdown_ok = True
        except Exception as stop_error:
            print(f"[FAIL] shutdown also failed: {stop_error}")
            shutdown_ok = False

    # ---- §12 report ----------------------------------------------------
    mean_valid = float(np.mean(valid_ratios)) if valid_ratios else 0.0
    depth_fps = depth_count / elapsed_s if elapsed_s > 0 else 0.0
    print("\n=============== PHASE 9 CHECKPOINT A REPORT ===============")
    print(f"wall time              : {elapsed_s:.1f}s")
    print(f"RGB frames             : {rgb_count}")
    print(f"depth frames           : {depth_count} ({depth_fps:.1f} fps)")
    print(f"first depth frame after: "
          f"{(first_depth_at - t_start):.1f}s" if first_depth_at else
          "first depth frame      : NEVER")
    print(f"max consecutive frames : {max_consecutive}")
    print(f"longest no-frame gap   : {longest_gap_ticks} ticks (~"
          f"{longest_gap_ticks / 30.0:.1f}s)")
    print(f"valid-depth ratio      : mean={mean_valid:.3f}"
          + (f" min={np.min(valid_ratios):.3f} max={np.max(valid_ratios):.3f}"
             if valid_ratios else ""))
    if frame_medians:
        print(f"depth stats (mm)       : "
              f"min={int(np.min(frame_mins))} "
              f"median={np.median(frame_medians):.0f} "
              f"max={int(np.max(frame_maxs))}")
    if center_medians_mm:
        med = np.array(center_medians_mm, dtype=np.float64)
        print(f"center region (m)      : median={np.median(med) / 1000:.2f} "
              f"std={np.std(med) / 1000:.3f} n={len(med)}")
    print(f"acquisition exceptions : {exceptions}")
    print(f"clean shutdown         : {'YES' if shutdown_ok else 'NO'}")
    print(f"depth health           : {camera.get_depth_health()}")

    # ---- Verdict (Checkpoint A evidence, §11) ---------------------------
    failures = []
    if rgb_count < 10:
        failures.append("RGB stream produced no frames (production baseline broken)")
    if depth_count == 0:
        failures.append("no depth frames received")
    else:
        if depth_fps < MIN_DEPTH_FPS:
            failures.append(f"depth fps {depth_fps:.1f} < {MIN_DEPTH_FPS}")
        if max_consecutive < MIN_CONSECUTIVE_FRAMES:
            failures.append(
                f"only {max_consecutive} consecutive depth frames "
                f"< {MIN_CONSECUTIVE_FRAMES} (stream not continuous)"
            )
        if mean_valid < MIN_VALID_RATIO:
            failures.append(
                f"mean valid-depth ratio {mean_valid:.3f} < {MIN_VALID_RATIO} "
                "(depth effectively empty)"
            )
    if not shutdown_ok:
        failures.append("shutdown did not complete cleanly")

    if failures:
        print("\nRESULT: FAIL — HARDWARE FUNCTIONALITY NOT VERIFIED")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nRESULT: PASS — CHECKPOINT A (HARDWARE-FUNCTIONALITY-VALIDATED)")
    print("NOTE: this validates hardware FUNCTIONALITY only.")
    print("      It does NOT validate measurement accuracy (Phase 10)")
    print("      and does NOT validate safety.")

    # Leave a parseable marker for the gate report.
    print("CHECKPOINT_A=PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
