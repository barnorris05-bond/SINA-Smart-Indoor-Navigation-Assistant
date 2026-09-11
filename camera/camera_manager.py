"""
camera_manager.py

Main camera abstraction for the Smart Indoor Navigation Assistant.
Using official DepthAI v3 API patterns (no 2.x artifacts: no XLinkOut,
no MonoCamera/ColorCamera nodes).

Phase 2: adds an aligned stereo-depth stream alongside the proven RGB
pipeline. The RGB path is unchanged; depth initialization failures
degrade gracefully to RGB-only operation (logged health state, never
a crash).
"""

import time
from typing import Any, Optional

import depthai as dai

from .rgb import RGBCamera
from .depth import DepthCamera
from config.camera import (
    RGB_WIDTH,
    RGB_HEIGHT,
    RGB_FPS,
    QUEUE_SIZE,
    BLOCKING_QUEUE,
    MONO_WIDTH,
    MONO_HEIGHT,
    DEPTH_FPS,
    ENABLE_LEFT_RIGHT_CHECK,
    ENABLE_STEREO_HARDWARE,
    STEREO_CONFIDENCE_THRESHOLD,
    DEPTH_WARMUP_FRAMES,
    get_stereo_preset,
    get_stereo_align_socket,
)
from utils.logger import logger

# ==========================================================
# Camera Status Constants
# ==========================================================

STATUS_DISCONNECTED = "DISCONNECTED"
STATUS_CONNECTING = "CONNECTING"
STATUS_CONNECTED = "CONNECTED"
STATUS_ERROR = "ERROR"

# Depth sub-status (health reporting; independent of RGB status).
DEPTH_DISABLED = "DEPTH_DISABLED"      # Not configured or failed to init.
DEPTH_WARMING_UP = "DEPTH_WARMING_UP"  # Configured, no usable frame yet.
DEPTH_OK = "DEPTH_OK"                  # Frames flowing.
DEPTH_STALE = "DEPTH_STALE"            # Frames flowed, then stopped.

# Seconds without a frame before working depth is declared stale.
DEPTH_STALE_AFTER_S = 2.0
# Warn when zero depth frames arrive after this many seconds (>= 3x the
# nominal warmup period of DEPTH_WARMUP_FRAMES at DEPTH_FPS).
DEPTH_WARMUP_WARN_S = max(10.0, 3.0 * DEPTH_WARMUP_FRAMES / DEPTH_FPS)


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
        self.depth_status: str = DEPTH_DISABLED

        # DepthAI v3 objects
        self.pipeline: Optional[dai.Pipeline] = None
        self.camera: Optional[dai.node.Camera] = None
        self.rgb_stream: Optional[Any] = None
        self.depth_stream: Optional[Any] = None
        self.stereo_node: Optional[dai.node.StereoDepth] = None

        # Sub-camera handlers
        self.rgb = RGBCamera()
        self.depth = DepthCamera()

        # Depth health telemetry
        self._depth_first_frame_at: Optional[float] = None
        self._depth_last_frame_at: Optional[float] = None
        self._depth_frame_count: int = 0
        self._depth_start_at: Optional[float] = None

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
            self._depth_start_at = time.time()
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
        self._log_depth_summary()

        self.pipeline = None
        self.camera = None
        self.rgb_stream = None
        self.depth_stream = None
        self.stereo_node = None

        self.status = STATUS_DISCONNECTED
        self.depth_status = DEPTH_DISABLED
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
        """Initialize RGB output stream using centralized config."""
        self.log("Initializing streams...")

        self.rgb_stream = self.camera.requestOutput(
            size=(RGB_WIDTH, RGB_HEIGHT),
            type=dai.ImgFrame.Type.BGR888p,
            fps=RGB_FPS
        )

        self.log("Creating MessageQueue...")

        rgb_queue = self.rgb_stream.createOutputQueue(
            maxSize=QUEUE_SIZE,
            blocking=BLOCKING_QUEUE
        )

        self.rgb.set_queue(rgb_queue)

        self.log(f"RGB stream initialized: {RGB_WIDTH}x{RGB_HEIGHT} @ {RGB_FPS} FPS")

        # Depth is ADDITIVE: a failure here must not take RGB down.
        try:
            self._initialize_depth_stream()
        except Exception as error:
            self.depth_status = DEPTH_DISABLED
            self.depth_stream = None
            self.stereo_node = None
            logger.warning(f"Stereo depth disabled (RGB continues): {error}")

    def _initialize_depth_stream(self) -> None:
        """
        Hardware gate: when ENABLE_STEREO_HARDWARE is False (current
        mock-depth development mode), the stereo branch is NOT built and
        the previously unstable stereo pipeline is never started. Depth
        stays DEPTH_DISABLED and RGB-only operation is the norm.
        """
        if not ENABLE_STEREO_HARDWARE:
            self.depth_status = DEPTH_DISABLED
            self.depth_stream = None
            self.stereo_node = None
            self.log("Stereo depth: DISABLED by config "
                     "(ENABLE_STEREO_HARDWARE=False; mock-depth mode)")
            return

        self._build_stereo_branch()

    def _build_stereo_branch(self) -> None:
        """
        Build the stereo-depth branch of the pipeline (DepthAI v3 API):

            CAM_B (mono) --GRAY8 640x400@15--\
                                              StereoDepth --> depth (RAW16, mm)
            CAM_C (mono) --GRAY8 640x400@15--/

        Depth is aligned to CAM_A so each depth pixel corresponds to an
        RGB pixel of the 640x400 frame (ready for Phase 3 fusion).
        """
        self.log("Initializing stereo depth...")

        pipeline = self.pipeline

        # Mono cameras for the stereo pair (v3 Camera node, CAM_B/CAM_C).
        left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        left_out = left.requestOutput(
            size=(MONO_WIDTH, MONO_HEIGHT),
            type=dai.ImgFrame.Type.GRAY8,
            fps=DEPTH_FPS,
        )
        right_out = right.requestOutput(
            size=(MONO_WIDTH, MONO_HEIGHT),
            type=dai.ImgFrame.Type.GRAY8,
            fps=DEPTH_FPS,
        )

        # StereoDepth node and configuration (v3 API only).
        stereo = pipeline.create(dai.node.StereoDepth)
        left_out.link(stereo.left)
        right_out.link(stereo.right)

        stereo.setDefaultProfilePreset(get_stereo_preset())
        stereo.setLeftRightCheck(ENABLE_LEFT_RIGHT_CHECK)
        # Align depth to the RGB camera: one depth value per RGB pixel.
        stereo.setDepthAlign(get_stereo_align_socket())

        # v3 pattern: confidence threshold lives on initialConfig, not the
        # node (setConfidenceThreshold is a DepthAI 2.x method).
        try:
            config = stereo.initialConfig
            config.costMatching.confidenceThreshold = STEREO_CONFIDENCE_THRESHOLD
            self.log(
                f"Stereo config: preset={get_stereo_preset().name} "
                f"LRC={ENABLE_LEFT_RIGHT_CHECK} "
                f"confThresh={STEREO_CONFIDENCE_THRESHOLD}"
            )
        except Exception as error:
            # Preset defaults remain usable if the config path differs.
            logger.warning(f"confidenceThreshold not applied: {error}")

        self.depth_stream = stereo.depth
        depth_queue = self.depth_stream.createOutputQueue(
            maxSize=QUEUE_SIZE,
            blocking=BLOCKING_QUEUE,
        )
        self.depth.set_queue(depth_queue)
        self.stereo_node = stereo

        self.depth_status = DEPTH_WARMING_UP
        self.log(
            f"Depth stream initialized: {MONO_WIDTH}x{MONO_HEIGHT} @ {DEPTH_FPS} FPS "
            f"(aligned to {get_stereo_align_socket()})"
        )

    # ======================================================
    # Frame Access
    # ======================================================

    def get_rgb_frame(self) -> Any:
        return self.rgb.get_frame()

    def get_depth_frame(self) -> Optional[Any]:
        """
        Latest depth frame as (H, W) uint16 millimeters, or None.

        Non-blocking: drains the depth queue to its newest frame so the
        caller never operates on stale depth, and returns None when no
        new frame is available this tick. Never raises on device-side
        hiccups; health is reported via get_depth_health().
        """
        if self.depth.queue is None:
            return None

        try:
            frame = self.depth.get_latest_frame()
        except Exception as error:
            logger.warning(f"Depth acquisition error: {error}")
            self._track_depth_health(False)
            return None

        if frame is None:
            self._track_depth_health(False)
            return None

        self._track_depth_health(True)
        return frame

    # ======================================================
    # Depth health telemetry
    # ======================================================

    def _track_depth_health(self, got_frame: bool) -> None:
        now = time.time()
        if got_frame:
            if self._depth_first_frame_at is None:
                self._depth_first_frame_at = now
                if self._depth_start_at is not None:
                    logger.info(
                        f"First depth frame after {now - self._depth_start_at:.1f}s"
                    )
            self._depth_last_frame_at = now
            self._depth_frame_count += 1
            if self.depth_status != DEPTH_OK:
                self.depth_status = DEPTH_OK
        else:
            if self.depth_status == DEPTH_OK:
                if (
                    self._depth_last_frame_at is not None
                    and now - self._depth_last_frame_at > DEPTH_STALE_AFTER_S
                ):
                    self.depth_status = DEPTH_STALE
                    logger.warning("Depth stream went stale (>2s without frames)")
            elif self.depth_status == DEPTH_WARMING_UP:
                if (
                    self._depth_start_at is not None
                    and now - self._depth_start_at > DEPTH_WARMUP_WARN_S
                ):
                    logger.warning("Depth stream: no frames since start (warmup)")

    def is_depth_available(self) -> bool:
        """True when usable depth frames are currently flowing."""
        return self.depth_status == DEPTH_OK

    def get_depth_health(self) -> dict:
        """Structured health snapshot for logging/UI/failure handling."""
        now = time.time()
        fps = None
        if (
            self._depth_first_frame_at is not None
            and self._depth_last_frame_at is not None
            and self._depth_last_frame_at > self._depth_first_frame_at
        ):
            fps = self._depth_frame_count / (
                self._depth_last_frame_at - self._depth_first_frame_at
            )
        return {
            "depth_status": self.depth_status,
            "frame_count": self._depth_frame_count,
            "measured_fps": fps,
            "seconds_since_last_frame": (
                now - self._depth_last_frame_at
                if self._depth_last_frame_at is not None else None
            ),
        }

    def _log_depth_summary(self) -> None:
        health = self.get_depth_health()
        if health["frame_count"] > 0:
            fps = health["measured_fps"]
            self.log(
                f"Depth summary: frames={health['frame_count']} "
                f"avg_fps={round(fps, 1) if fps else '?'} "
                f"status={health['depth_status']}"
            )
        else:
            self.log(f"Depth summary: no frames received ({health['depth_status']})")

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
        return f"CameraManager(status='{self.status}', depth='{self.depth_status}')"