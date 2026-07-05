"""
pipeline_builder.py

Responsible for constructing all
DepthAI processing pipelines.
"""

import depthai as dai


class PipelineBuilder:

    def __init__(self):
        self.pipeline = dai.Pipeline()

    def build_rgb_pipeline(self):
        """
        Build the RGB camera pipeline.
        """

        raise NotImplementedError