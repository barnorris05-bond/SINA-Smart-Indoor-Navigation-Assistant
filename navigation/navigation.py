"""
config/navigation.py
Central static configuration repository for navigation policies and labels.
"""

from typing import Set, Dict

# Semantic classification of architectural and environmental hazards
NAVIGATION_OBSTACLES: Set[str] = {
    "person",
    "chair",
    "table",
    "door",
    "stairs",
    "wall",
    "wheelchair",
    "dog",
    "child",
    "stroller"
}

# Base priority values (navigation importance weights)
OBJECT_PRIORITIES: Dict[str, int] = {
    "person": 100,
    "child": 100,
    "stroller": 95,
    "wheelchair": 95,
    "chair": 90,
    "door": 90,
    "stairs": 100,
    "table": 80,
    "dog": 85,
    "bottle": 40,
    "book": 20,
    "laptop": 10,
    "mouse": 30,
    "keyboard": 25,
}

# Decision-making thresholds used by the policy engine
CRITICAL_PRIORITY_THRESHOLD: int = 80
