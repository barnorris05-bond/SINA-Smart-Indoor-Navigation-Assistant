"""
camera_manager.py

Main camera abstraction for the Smart Indoor Navigation Assistant.
Using official DepthAI v3 API patterns.
"""

import depthai as dai
from .rgb import RGBCamera
from .depth import DepthCamera
from typing import Any, Optional

# ==========================================================
# Camera Status Constants
# ==========================================================

STATUS_DISCONNECTED = "DISCONNECTED"
STATUS_CONNECTING = "CONNECTING"
STATUS_CONNECTED = "CONNECTED"
STATUS_ERROR = "ERROR"


class CameraManager:
    """
    CameraManager is the only class that communicates directly
    with the OAK-D Lite using official DepthAI v3 API.
    """

    def __init__(self) -> None:
        """
        Initialize the Camera Manager.
        """
        # Camera state
        self.status: str = STATUS_DISCONNECTED

        # DepthAI v3 objects
        self.pipeline: Optional[dai.Pipeline] = None
        self.camera: Optional[dai.node.Camera] = None
        self.rgb_stream: Optional[Any] = None
        self.depth_stream: Optional[Any] = None

        # Sub-camera handlers
        self.rgb = RGBCamera()
        self.depth = DepthCamera()

    # ======================================================
    # Camera Lifecycle
    # ======================================================

    def start(self) -> None:
        """
        Start the camera using official DepthAI v3 flow.
        """
        if self.status == STATUS_CONNECTED:
            self.log("Camera is already running.")
            return

        self.status = STATUS_CONNECTING

        try:
            self.log("Connecting to OAK-D Lite...")

            # Correct order: Create → Configure queues → Start
            self._create_pipeline()
            self._initialize_streams()
            self._start_pipeline()

            self.status = STATUS_CONNECTED
            self.log("Camera connected successfully.")

        except Exception as error:
            self.status = STATUS_ERROR
            self.log(f"Camera startup failed: {error}")
            raise

    def stop(self) -> None:
        """
        Stop the camera and release resources.
        """
        if self.status == STATUS_DISCONNECTED:
            self.log("Camera is already stopped.")
            return

        self.log("Stopping camera...")

        self.pipeline = None
        self.camera = None
        self.rgb_stream = None
        self.depth_stream = None

        self.status = STATUS_DISCONNECTED
        self.log("Camera disconnected.")

    def reset(self) -> None:
        """Reset the camera manager."""
        self.log("Resetting camera manager...")
        self.stop()

    def _create_pipeline(self) -> None:
        """Create the DepthAI v3 pipeline with Camera node."""
        self.log("Creating pipeline...")
        self.pipeline = dai.Pipeline()

        self.camera = self.pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A
        )

    def _start_pipeline(self) -> None:
        """Start the pipeline (boots the device)."""
        self.log("Starting pipeline...")
        if self.pipeline:
            self.pipeline.start()

    def _initialize_streams(self) -> None:
        """Initialize RGB output stream using official DepthAI v3 API."""
        self.log("Initializing streams...")

        self.rgb_stream = self.camera.requestOutput(
            size=(1280, 720),
            type=dai.ImgFrame.Type.BGR888p,
            fps=30
        )

        # Diagnostic log to confirm this code is running
        self.log("Creating MessageQueue...")

        rgb_queue = self.rgb_stream.createOutputQueue(
            maxSize=4,
            blocking=False
        )

        self.rgb.set_queue(rgb_queue)

        self.log("RGB stream initialized (low-latency mode).")

    # ======================================================
    # Frame Access
    # ======================================================

    def get_rgb_frame(self) -> Any:
        return self.rgb.get_frame()

    def get_depth_frame(self) -> Any:
        return self.depth.get_frame()

    # ======================================================
    # Status
    # ======================================================

    def is_connected(self) -> bool:
        return self.status == STATUS_CONNECTED

    def get_status(self) -> str:
        return self.status

    # ======================================================
    # Utilities
    # ======================================================

    def log(self, message: str) -> None:
        print(f"[CameraManager] {message}")

    def __repr__(self) -> str:
        return f"CameraManager(status='{self.status}')"