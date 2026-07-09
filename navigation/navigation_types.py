"""
navigation/navigation_types.py
Strongly-typed schemas and primitives for safety policies.
"""

from enum import Enum, auto


class NavigationAction(Enum):
    """Enumeration of safety commands sent to the user interface."""
    CONTINUE = auto()
    SLOW_DOWN = auto()
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    STOP = auto()
