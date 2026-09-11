"""
config/depth_fusion.py

Object-distance fusion configuration (Phases B-D).

Small by design: fusion has exactly two tunables. The distance VALUES
used in mock mode live in config/mock_depth.py; navigation thresholds
live in config/navigation.py.

!!! Development/test values — recalibrate with real stereo (Phase E/F) !!!
"""

# A measurement older than this (seconds, per DepthMeasurement.timestamp)
# is treated as STALE by the fusion layer even if the provider forgot to
# mark it — defense in depth so old values never silently drive decisions.
# Real stereo runs at ~15 FPS, so anything >2s old is unusably old.
MEASUREMENT_STALE_S = 2.0

# DEVELOPMENT/TEST ONLY: allow SIMULATED distances to flow into
# navigation decisions. Must be False once real depth is validated,
# or mock values could contaminate a live assistive system.
ALLOW_SIMULATED_FOR_NAVIGATION = True
