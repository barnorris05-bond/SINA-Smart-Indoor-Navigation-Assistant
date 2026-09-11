"""
tests/test_depth_stream.py

PHASE 2 HARDWARE TEST: RGB + aligned stereo depth working together.

Runs the PRODUCTION path (CameraManager with both streams in one
DepthAI 3.x pipeline) and measures:

  - RGB frames still flowing (regression guard for Phase 1 baseline)
  - depth warmup time (start -> first frame)
  - depth frame dimensions, dtype
  - depth FPS over a measurement window
  - frame validity ratio (non-zero, in-range pixels)
  - center-region distance stability (median/std of per-frame medians)

Exit codes:
  0 = PASS (valid depth received and measured)
  1 = FAIL (device present but depth did not meet criteria)
  2 = NO DEVICE (OAK-D not connected — not a code failure)

Usage:
  .venv\\Scripts\\python.exe tests\\test_depth_stream.py [--seconds 10] [--preview]

--preview shows the RGB frame with the live center distance (press q to
quit early). Without it, the test runs headless and prints measurements.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import depthai as dai

# This is the Phase E HARDWARE test: it must exercise the real stereo
# branch regardless of the development default (ENABLE_STEREO_HARDWARE
# is False during mock-depth development). Set before importing
# CameraManager.
import config.camera as _camcfg
_camcfg.ENABLE_STEREO_HARDWARE = True

from camera.camera_manager import CameraManager
from depth.validity import validate_depth_frame, measure_region


def main() -> int:
    args = sys.argv[1:]
    measure_seconds = 10.0
    if "--seconds" in args:
        measure_seconds = float(args[args.index("--seconds") + 1])
    show_preview = "--preview" in args

    # ---- Pre-flight: device presence (distinct exit code) --------------
    try:
        devices = dai.Device.getAllAvailableDevices()
    except Exception as error:
        print(f"[preflight] device query failed: {error}")
        return 2
    if not devices:
        print("[preflight] NO OAK-D DEVICE FOUND.")
        print("            Connect the OAK-D Lite and re-run this test.")
        return 2
    print(f"[preflight] device(s): {[d.deviceId for d in devices]}")

    camera = CameraManager()
    try:
        camera.start()
    except Exception as error:
        print(f"[FAIL] camera start failed: {error}")
        return 1

    center_bbox = (270, 150, 370, 250)  # central 100x100 of 640x400
    center_medians_mm = []

    rgb_count = 0
    depth_count = 0
    valid_ratios = []
    t_start = time.time()
    first_depth_at = None
    t_end = t_start + measure_seconds + 5.0  # grace for warmup

    print(f"[test] streaming for ~{measure_seconds:.0f}s (plus warmup)...")

    try:
        if camera.is_connected():
            print("[test] camera connected:", repr(camera))

        while time.time() < t_end:
            loop_t0 = time.perf_counter()

            frame = camera.get_rgb_frame()
            if frame is not None:
                rgb_count += 1

            depth = camera.get_depth_frame()
            if depth is not None:
                depth_count += 1
                if first_depth_at is None:
                    first_depth_at = time.time()
                    print(
                        f"[test] first depth frame after "
                        f"{first_depth_at - t_start:.1f}s | "
                        f"shape={depth.shape} dtype={depth.dtype}"
                    )
                report = validate_depth_frame(depth)
                valid_ratios.append(report["valid_ratio"])
                stats = measure_region(depth, center_bbox)
                if stats.valid:
                    center_medians_mm.append(stats.distance_mm)

            if show_preview and frame is not None:
                import cv2
                if center_medians_mm:
                    d_m = center_medians_mm[-1] / 1000.0
                    cv2.putText(
                        frame, f"CENTER: {d_m:.2f} m",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA,
                    )
                health = camera.get_depth_health()
                cv2.putText(
                    frame,
                    f"depth: {health['depth_status']} f={health['frame_count']}",
                    (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2,
                )
                cv2.imshow("SINA depth stream test (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Pace roughly to the processing loop's natural rate.
            elapsed = time.perf_counter() - loop_t0
            time.sleep(max(0.0, 1.0 / 30.0 - elapsed))

        elapsed_s = time.time() - t_start
    finally:
        if show_preview:
            import cv2
            cv2.destroyAllWindows()
        camera.stop()

    # ---- Report ---------------------------------------------------------
    print("\n================ PHASE 2 HARDWARE TEST REPORT ================")
    print(f"wall time            : {elapsed_s:.1f}s")
    print(f"RGB frames           : {rgb_count} ({rgb_count / elapsed_s:.1f} fps)")
    print(f"depth frames         : {depth_count} ({depth_count / elapsed_s:.1f} fps)")
    print(f"depth warmup         : "
          f"{(first_depth_at - t_start):.1f}s" if first_depth_at else
          "depth warmup         : NO FRAMES")
    if valid_ratios:
        print(f"validity ratio       : mean={np.mean(valid_ratios):.3f} "
              f"min={np.min(valid_ratios):.3f} max={np.max(valid_ratios):.3f}")
    if center_medians_mm:
        med = np.array(center_medians_mm, dtype=np.float64)
        print(f"center distance (m)  : median={np.median(med) / 1000:.2f} "
              f"std={np.std(med) / 1000:.3f} n={len(med)}")
    health = camera.get_depth_health()
    print(f"depth health         : {health}")

    # ---- Verdict ----------------------------------------------------------
    failures = []
    if rgb_count < 10:
        failures.append("RGB stream did not produce frames (baseline broken!)")
    if depth_count == 0:
        failures.append("no depth frames received")
    elif depth_count / elapsed_s < 8.0:
        failures.append(
            f"depth fps {depth_count / elapsed_s:.1f} below 8.0 (target ~15)"
        )
    if valid_ratios and np.mean(valid_ratios) < 0.05:
        failures.append(
            f"mean validity ratio {np.mean(valid_ratios):.3f} < 0.05 "
            "(depth effectively empty)"
        )

    if failures:
        print("\nRESULT: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nRESULT: PASS — RGB + aligned stereo depth are streaming reliably.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
