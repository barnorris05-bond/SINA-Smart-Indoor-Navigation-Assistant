"""
config/navigation_distance.py

DISTANCE-AWARE NAVIGATION THRESHOLDS (Phase D).

!!! DEVELOPMENT/TEST THRESHOLDS — NOT CALIBRATED !!!

These values drive development and tests with SIMULATED mock depth.
They MUST be re-calibrated and validated against real OAK-D stereo
measurements (Phase E/F) before any real-world use. Do not treat them
as final safety limits.

Policy shape (replaces pure spatial rules only when usable distance
exists; otherwise the legacy spatial Navigator decides unchanged):

    distance <= CRITICAL_DISTANCE_M        -> STOP
    distance <  WARNING_DISTANCE_M          -> center: SLOW_DOWN
                                               lateral: move away
    distance >= FAR_DISTANCE_M              -> CONTINUE (obstacle far)
"""

# At or below this distance, stop immediately (any region).
# DEV/TEST value derived from the reference scenario "person at 1.2m
# -> STOP"; recalibrate against real stereo measurements (Phase E/F).
CRITICAL_DISTANCE_M = 1.2

# At or below this distance an obstacle warrants a reaction
# (slow/avoid). Inclusive so the reference scenario "chair at 2.5m
# -> SLOW_DOWN" holds. DEV/TEST value — recalibrate later.
WARNING_DISTANCE_M = 2.5

# At or beyond this distance an obstacle no longer alters behavior.
FAR_DISTANCE_M = 4.0

# When several objects qualify at the same policy step, break ties by
# distance first, then priority (higher wins).
TIEBREAK_BY_PRIORITY = True
