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

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Non-blocking: drain the queue and return the NEWEST depth frame
        as an (H, W) uint16 array (millimeters), or None when no new
        frame is available this tick.

        Draining (instead of taking one frame) guarantees the caller
        never operates on stale depth while the vision loop runs at its
        own cadence.
        """
        if self.queue is None:
            return None

        latest = None
        try:
            packet = self.queue.tryGet()
            while packet is not None:
                latest = packet         # keep the freshest frame
                packet = self.queue.tryGet()
        except Exception:
            # Device-side hiccups must not crash the vision loop.
            return None

        if latest is None:
            return None

        frame = latest.getFrame()

        # Normalize shape: some paths yield (1, H, W) — reduce to (H, W).
        if hasattr(frame, "ndim") and frame.ndim == 3 and frame.shape[0] == 1:
            frame = frame[0]

        return frame

    def is_ready(self) -> bool:
        """Check whether the depth stream is available."""
        return self.queue is not None