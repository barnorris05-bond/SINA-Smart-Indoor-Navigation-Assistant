"""
utils/fps.py
Simple FPS counter for real-time display.
"""

import time


class FPS:
    def __init__(self):
        self.prev_time = time.time()
        self.frame_count = 0
        self.fps = 0

    def update(self) -> float:
        """Update FPS and return current value."""
        self.frame_count += 1
        current_time = time.time()

        # Update FPS every second
        if current_time - self.prev_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.prev_time = current_time

        return self.fps

    def reset(self):
        """Reset the counter."""
        self.prev_time = time.time()
        self.frame_count = 0
        self.fps = 0