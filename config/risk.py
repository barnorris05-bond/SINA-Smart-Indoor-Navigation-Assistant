"""
config/risk.py

RISK ESTIMATION CONFIGURATION (Phase 7).

!!! DEVELOPMENT/TEST THRESHOLDS — NOT CALIBRATED !!!

Chosen for the current development setup with SIMULATED mock depth.
NOT derived from real-world evaluation and NOT safety-calibrated.
Recalibrate against real stereo + measured motion (Phases 9-10)
before drawing any conclusion from these numbers.

ALIGNMENT: distance bands (CRITICAL/WARNING/FAR) and the high-priority
threshold are NOT redefined here - they are imported directly from
config/navigation_distance.py and config/navigation.py by the risk
estimator, so risk and navigation can never silently diverge. Only
risk-native constants live in this module.

TTC policy: TTC ~= distance / closing_rate is computed ONLY when the
motion layer reports a hysteresis-confirmed APPROACHING state with a
rate of at least TTC_MIN_CLOSING_RATE_MPS. Otherwise TTC is UNKNOWN
and risk falls back to the distance/priority ladder. A meaningless TTC
must never be produced from stale, noisy or receding data.
"""

# TTC at/below which risk is CRITICAL (seconds). Dev/test value: at
# ~1.5 s a healthy adult walking pace covers ~1.7 m - roughly the
# critical distance band. Recalibrate with measured motion.
TTC_CRITICAL_S = 1.5

# TTC at/below which risk is HIGH (seconds). Dev/test value.
TTC_WARNING_S = 3.0

# Minimum closing rate (m/s) for TTC to be considered trustworthy.
# Below this, depth noise dominates and the quotient is meaningless.
TTC_MIN_CLOSING_RATE_MPS = 0.1

# Anti-flap: a risk level can rise instantly but may only DROP after
# the estimator has produced the lower level on this many consecutive
# assessments for the same track. Prevents CRITICAL -> LOW from one
# noisy frame while keeping escalation immediate.
RISK_DROP_CONFIRM_FRAMES = 3
