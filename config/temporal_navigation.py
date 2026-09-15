"""
config/temporal_navigation.py

TEMPORAL NAVIGATION STABILIZATION CONFIGURATION (Phase 8).

!!! DEVELOPMENT/TEST VALUES — NOT CALIBRATED !!!

Chosen to make the stabilized decision stream behave sensibly under
SIMULATED mock depth at 15 FPS. NOT derived from real-world evaluation
and NOT safety-calibrated. Recalibrate with measured stereo, motion and
risk data (Phases 9-10) before drawing any safety conclusion.

POLICY (TemporalNavigator):

- ESCALATION IS IMMEDIATE: a candidate decision that is more severe
  than the current one, or any STOP candidate while the scene is not
  already critical, is accepted on the frame it first appears.
- DE-ESCALATION REQUIRES PERSISTENCE: a candidate less severe than the
  current decision must appear on DEESCALATION_CONFIRM_FRAMES
  consecutive frames before the displayed action relaxes. One noisy
  low-risk frame can never erase a confirmed STOP.
- ACTION CHANGES AT EQUAL SEVERITY REQUIRE PERSISTENCE: any switch
  between two different actions of the same severity (MOVE_LEFT <->
  MOVE_RIGHT direction flips, and forward <-> lateral changes such as
  SLOW_DOWN <-> MOVE_*) requires ACTION_CHANGE_CONFIRM_FRAMES
  consecutive candidates for the new action, preventing directional
  chatter from noisy detections while still allowing a genuine
  persistent change to take effect.
- CRITICAL OVERRIDE: when True, a STOP candidate is never delayed by
  any confirmation counter. A genuine emergency is never smoothed.
- SCENE-CRITICAL FORCES STOP: when True, a scene whose maximum
  annotated risk is CRITICAL produces a STOP decision even when the
  distance-band navigator's candidate was weaker (e.g. a fast
  approach at 2.6 m can be risk-CRITICAL by TTC while still outside
  the distance STOP band). This is the Phase 8 point where risk
  annotation starts to influence navigation; RiskEstimator remains
  the sole authority for CLASSIFYING risk.
- MINIMUM ACTION DURATION is deliberately NOT a separate knob: it is
  subsumed by the confirmation counters above (a 1-frame flapping
  candidate can never be displayed), and STOP is exempt by
  construction because STOP always escalates immediately.
"""

# Consecutive lower-severity candidates required before the displayed
# action relaxes. Aligned with the risk layer's anti-flap window
# (config/risk.py RISK_DROP_CONFIRM_FRAMES = 3) so navigation never
# de-escalates faster than the risk estimate that feeds it.
DEESCALATION_CONFIRM_FRAMES = 3

# Consecutive candidates required before a switch to a DIFFERENT
# action at equal severity is accepted (covers MOVE_LEFT <-> MOVE_RIGHT
# direction flips and forward <-> lateral changes alike). A single
# noisy flip is suppressed; two in a row (a persistent change) switch.
ACTION_CHANGE_CONFIRM_FRAMES = 2

# STOP candidates bypass all confirmation (safety override): a STOP is
# never delayed by de-escalation or action-change smoothing.
CRITICAL_OVERRIDE = True

# A scene whose maximum annotated risk is CRITICAL forces the displayed
# decision to STOP even when the candidate navigator chose a weaker
# action (RiskEstimator classifies; this layer maps CRITICAL -> STOP).
CRITICAL_RISK_FORCE_STOP = True
