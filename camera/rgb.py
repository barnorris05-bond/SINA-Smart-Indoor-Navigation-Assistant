"""
rgb.py

Handles RGB frame retrieval from the OAK-D Lite using DepthAI v3.
"""

import depthai as dai
import numpy as np
from typing import Optional


class RGBCamera:
    """
    Handles retrieval of RGB frames from the OAK-D Lite.
    """

    def __init__(self) -> None:
        self.queue: Optional[dai.MessageQueue] = None

    def set_queue(self, queue: dai.MessageQueue) -> None:
        """Assign the RGB output queue from CameraManager."""
        self.queue = queue

    def get_frame(self) -> np.ndarray:
        """
        Retrieve the latest RGB frame as a NumPy array (BGR format).

        Note: queue.get() is blocking, so it waits for a frame.
        """
        if self.queue is None:
            raise RuntimeError("RGB output queue has not been initialized.")

        frame: dai.ImgFrame = self.queue.get()
        return frame.getCvFrame()

    def is_ready(self) -> bool:
        """Check whether the RGB stream is available."""
        return self.queue is not None