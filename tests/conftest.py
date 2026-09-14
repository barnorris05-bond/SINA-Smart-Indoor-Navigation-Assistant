"""
tests/conftest.py

Pytest collection policy.

tests/ mixes two kinds of scripts:

1. HARDWARE-FREE logic suites (pytest-safe):
     test_audio_logic.py
     test_depth_validity.py
     test_mock_depth_stack.py
     test_pipeline_mock_e2e.py

2. MANUAL HARDWARE probes (test_*.py names, but they build DepthAI
   pipelines / open devices at import time — and test_depth_stream.py
   flips ENABLE_STEREO_HARDWARE=True on import). A plain `pytest` must
   NEVER collect these: they would crash CI-style runs, and worse,
   silently start the previously unstable stereo pipeline.

Run hardware probes deliberately and individually, e.g.:
    ./.venv/Scripts/python.exe tests/test_depth_stream.py --preview
"""

import os

# Only these files are collected by `pytest` / `python -m pytest -q`.
COLLECTABLE = {
    "test_audio_logic.py",
    "test_depth_validity.py",
    "test_mock_depth_stack.py",
    "test_object_tracker.py",
    "test_motion_estimator.py",
    "test_risk_estimator.py",
    "test_temporal_navigator.py",
    "test_oak_provider.py",
    "test_evaluation_infra.py",
    "test_evaluation_analysis.py",
    "test_calibration_framework.py",
    "test_capture_protocol.py",
    "test_synthetic_calibration_campaign.py",
    "test_pipeline_mock_e2e.py",
}


def pytest_ignore_collect(collection_path, config):
    """
    Allow-list policy: collect ONLY the hardware-free suites.

    This directory contains hardware probes matching both `test_*.py`
    and `*_test.py` pytest patterns (e.g. simple_device_test.py opens
    dai.Device() at import), so anything not explicitly listed is
    ignored.
    """
    name = os.path.basename(str(collection_path))
    if name.endswith(".py") and name not in COLLECTABLE and name != "conftest.py":
        return True
    return None
