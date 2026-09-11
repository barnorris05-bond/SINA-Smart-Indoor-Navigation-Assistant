"""
audio/message_builder.py

Pure functions: NavigationDecision -> (text, state_signature).

No timing logic, no engine, no side effects — fully unit-testable.
The state signature identifies *what* is being announced so the
Announcer can deduplicate identical consecutive events.
"""

from typing import Optional, Tuple

from config.audio import (
    MESSAGES,
    MESSAGE_DEFAULT,
    DISTANCE_BUCKET_NEAR,
    DISTANCE_BUCKET_MID,
    DISTANCE_ROUNDING,
)
from navigation.navigation_types import NavigationAction


def distance_bucket(distance_m: Optional[float]) -> str:
    """Bucket a distance for state-signature jitter immunity."""
    if distance_m is None:
        return "unknown"
    if distance_m < DISTANCE_BUCKET_NEAR:
        return "near"
    if distance_m < DISTANCE_BUCKET_MID:
        return "mid"
    return "far"


def round_distance(distance_m: Optional[float]) -> Optional[float]:
    """Round a distance to the configured anti-jitter step (e.g. 0.5 m)."""
    if distance_m is None:
        return None
    step = max(DISTANCE_ROUNDING, 1e-6)
    return round(distance_m / step) * step


def format_distance(distance_m: Optional[float]) -> str:
    """Human clause for a distance, or '' when no distance is known."""
    rounded = round_distance(distance_m)
    if rounded is None or rounded <= 0:
        return ""
    # Snap e.g. 3.0 -> "3", 1.5 -> "1.5"; singular/plural handled.
    text = f"{rounded:.1f}".rstrip("0").rstrip(".")
    unit = "meter" if abs(rounded - 1.0) < 1e-9 else "meters"
    return f" at {text} {unit}"


def build_message(decision) -> Tuple[str, str]:
    """
    Build the spoken text and state signature for a NavigationDecision.

    Returns (text, signature) where signature is
    "ACTION|label|region|bucket" — identical scenes collapse to the
    identical signature so the Announcer suppresses repetition.
    """
    action = decision.action.name
    label = decision.trigger.label.lower() if decision.trigger else ""
    region = decision.trigger.region if decision.trigger else ""

    # Only lateral moves mention the side; forward threats get "ahead".
    dist_text = format_distance(decision.trigger.distance) if decision.trigger else ""

    template = MESSAGES.get(action, MESSAGE_DEFAULT)
    display_label = label.capitalize() if label else "Obstacle"

    text = template.format(label=display_label, dist=dist_text)

    signature = f"{action}|{label}|{region}|{distance_bucket(decision.trigger.distance) if decision.trigger else 'none'}"
    return text, signature
