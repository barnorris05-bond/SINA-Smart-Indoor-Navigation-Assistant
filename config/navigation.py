"""
config/navigation.py

Navigation configuration shared across the project.
"""

# Ignore detections below this confidence
MIN_CONFIDENCE = 0.50

# Object importance for navigation
OBJECT_PRIORITIES = {
    "person": 100,
    "stairs": 100,
    "chair": 90,
    "door": 90,
    "table": 80,
    "bottle": 40,
    "mouse": 30,
    "keyboard": 25,
    "book": 20,
    "laptop": 10,
}

# Objects that should affect navigation decisions
NAVIGATION_OBSTACLES = {
    "person",
    "chair",
    "door",
    "stairs",
    "table",
}

# Policy evaluation threshold
CRITICAL_PRIORITY_THRESHOLD = 80
