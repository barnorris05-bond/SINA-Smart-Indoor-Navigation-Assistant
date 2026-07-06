"""
camera/stereo_pipeline.py
Minimal stereo depth pipeline for debugging.
"""

import depthai as dai
from config.camera import MONO_WIDTH, MONO_HEIGHT


class StereoPipelineBuilder:
    """
    Builds a minimal stereo depth pipeline (depth only).
    """

    @staticmethod
    def create():
        """
        Create pipeline with only stereo depth.
        Returns (pipeline, depth_output)
        """
        pipeline = dai.Pipeline()

        # Left and Right Mono Cameras
        left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        # Stereo Depth
        stereo = pipeline.create(dai.node.StereoDepth)
        left.requestOutput((MONO_WIDTH, MONO_HEIGHT), dai.ImgFrame.Type.GRAY8).link(stereo.left)
        right.requestOutput((MONO_WIDTH, MONO_HEIGHT), dai.ImgFrame.Type.GRAY8).link(stereo.right)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
        stereo.setLeftRightCheck(True)

        return pipeline, stereo.depth