import depthai as dai

print("=" * 60)
print("Testing DepthAI Queue Lifecycle")
print("=" * 60)

pipeline = dai.Pipeline()

print("Creating Camera node...")
cam = pipeline.create(dai.node.Camera).build(
    dai.CameraBoardSocket.CAM_A
)

print("Requesting RGB output...")
rgb_output = cam.requestOutput(
    size=(1280, 720),
    type=dai.ImgFrame.Type.BGR888p,
    fps=30,
)

print("Creating output queue...")
queue = rgb_output.createOutputQueue(
    maxSize=4,
    blocking=False,
)

print("✓ Queue created successfully")

print("Starting pipeline...")
pipeline.start()

print("✓ Pipeline started")

print("Waiting for first frame...")

frame = queue.get()

print("✓ Frame received!")

cv_frame = frame.getCvFrame()

print("Frame shape:", cv_frame.shape)