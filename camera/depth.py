"""
depth.py

Handles depth frame retrieval from the OAK-D Lite using DepthAI v3.
"""

import depthai as dai
import numpy as np
from typing import Optional


class DepthCamera:
    """
    Handles retrieval of depth frames from the OAK-D Lite.
    """

    def __init__(self) -> None:
        self.queue: Optional[dai.MessageQueue] = None

    def set_queue(self, queue: dai.MessageQueue) -> None:
        """Assign the depth output queue from CameraManager."""
        self.queue = queue

    def get_frame(self) -> np.ndarray:
        """
        Retrieve the latest depth frame as a NumPy array.

        Note: queue.get() is blocking.
        """
        if self.queue is None:
            raise RuntimeError("Depth output queue has not been initialized.")

        packet: dai.ImgFrame = self.queue.get()
        return packet.getFrame()        # Raw depth (uint16)

    def is_ready(self) -> bool:
        """Check whether the depth stream is available."""
        return self.queue is not None