"""
config/motion.py

MOTION / APPROACH ESTIMATION CONFIGURATION (Phase 6).

!!! DEVELOPMENT/TEST TUNING - NOT CALIBRATED !!!

Chosen for the current development setup (640x400 RGB @ 15 FPS, mock
depth ~15 Hz, indoor ranges). NOT derived from real-world evaluation
and NOT safety-calibrated. Recalibrate against real stereo sequences
(Phases 9-10) before drawing any conclusion from these numbers.
"""

# Rolling-median window (samples) applied to each track's distance
# history before rate estimation. A median rejects single-frame depth
# outliers (speckle, LRC failures, reflective-surface glitches) far
# better than a mean of the same size, and it is fully deterministic.
# Window 5 ~= 0.33 s of history at 15 FPS.
MOTION_MEDIAN_WINDOW = 5

# Classification needs at least this many usable samples...
MIN_SAMPLES = 3

# ...spanning at least this many seconds. Two samples 30 ms apart
# cannot distinguish a 0.1 m/s approach from depth noise.
MIN_OBSERVATION_S = 0.5

# Rate estimation uses only samples inside the most recent span of this
# many seconds (count-capped by HISTORY_MAXLEN). A long detection gap
# (e.g. stale depth) therefore never blends pre-gap samples into a
# post-gap rate: after a >WINDOW_SPAN_S gap the estimator honestly
# returns UNKNOWN until fresh evidence re-accumulates.
WINDOW_SPAN_S = 3.0

# Per-track history cap (bounded memory; oldest samples dropped first).
HISTORY_MAXLEN = 30

# A track absent from detections (or absent valid samples) for more
# than this many consecutive estimator frames has its motion history
# purged. The tracker expires tracks at 10 missed frames; the margin
# here covers tracks that linger just past tracker expiry.
MAX_HISTORY_AGE_FRAMES = 15

# Rate thresholds in m/s (positive = distance decreasing = approaching).
# HYSTERESIS: the three bands leave dead zones between them. A track
# switches INTO a state only when the estimated closing rate is inside
# that state's band; rates inside a dead zone KEEP the previous state.
# This suppresses classification flicker from noisy rate estimates.
#   [0.05 .. 0.15] m/s  -> dead zone (state unchanged)
#   [-0.15 .. -0.05]    -> dead zone (state unchanged)
APPROACH_RATE_MPS = 0.15    # switch INTO APPROACHING at/above this
RECEDE_RATE_MPS = 0.15      # switch INTO RECEDING at/below -this
STATIONARY_RATE_MPS = 0.05  # switch INTO STATIONARY within +/-this
