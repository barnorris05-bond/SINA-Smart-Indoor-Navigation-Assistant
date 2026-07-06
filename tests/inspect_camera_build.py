import depthai as dai

pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.Camera)

print("=" * 60)
print("Camera.build Inspection")
print("=" * 60)

help(cam.build)