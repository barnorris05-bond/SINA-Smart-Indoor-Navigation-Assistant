"""
config/audio.py

Centralized configuration for the audio guidance subsystem (Phase 1).

Every tunable lives here — no magic numbers inside audio/ modules.
"""

# ==========================================================
# TTS Engine
# ==========================================================

TTS_RATE = 175          # Words per minute. Default 200 is too fast for
                        # assistive guidance; 175 favors clarity.
TTS_VOLUME = 1.0        # 0.0 - 1.0

# ==========================================================
# Cooldowns (seconds) per navigation action.
# Prevents repeating the same announcement every frame.
# ==========================================================

COOLDOWNS = {
    "STOP": 1.0,        # Safety-critical: may repeat relatively quickly.
    "SLOW_DOWN": 3.0,
    "MOVE_LEFT": 3.0,
    "MOVE_RIGHT": 3.0,
    "CONTINUE": 5.0,    # Non-critical: suppressed aggressively (anti-chatter).
}
COOLDOWN_DEFAULT = 3.0

# ==========================================================
# Announcement priority ranking (higher = more critical).
# Used to decide whether a state change may interrupt an
# active cooldown (safety escalation).
# ==========================================================

PRIORITY_MAP = {
    "STOP": 4,
    "SLOW_DOWN": 3,
    "MOVE_LEFT": 2,
    "MOVE_RIGHT": 2,
    "CONTINUE": 1,
}
PRIORITY_DEFAULT = 1

# A new announcement is an "escalation" (bypasses cooldown) when its
# priority exceeds the last announced priority by at least this amount.
ESCALATION_DELTA = 1

# ==========================================================
# Distance anti-jitter (used from Phase 4 onward, wired now).
# ==========================================================

DISTANCE_BUCKET_NEAR = 1.5    # meters: below -> "near"
DISTANCE_BUCKET_MID = 3.0     # meters: below -> "mid", else -> "far"
DISTANCE_ROUNDING = 0.5       # meters: distances rounded to this step

# Same-state safety escalation: if the same obstacle came at least this
# much closer since the last announcement, re-announce immediately.
ESCALATION_DISTANCE_DELTA = 0.5

# ==========================================================
# Message templates.
# {label}  -> "Person", "Chair", ... or "Obstacle" when unknown
# {dist}   -> optional distance clause ("" when no distance known)
# ==========================================================

MESSAGES = {
    "STOP": "{label}{dist} ahead. Please stop.",
    "SLOW_DOWN": "{label}{dist} ahead. Slow down.",
    "MOVE_LEFT": "Obstacle on your right{dist}. Move left.",
    "MOVE_RIGHT": "Obstacle on your left{dist}. Move right.",
    "CONTINUE": "Path clear. Continue forward.",
}
MESSAGE_DEFAULT = "Obstacle{dist} ahead. Please stop."

# ==========================================================
# Speech worker
# ==========================================================

SPEECH_QUEUE_SIZE = 16      # Max queued utterances; overflow is dropped.
SPEAKER_SHUTDOWN_TIMEOUT = 3.0  # seconds to wait for the TTS thread.
