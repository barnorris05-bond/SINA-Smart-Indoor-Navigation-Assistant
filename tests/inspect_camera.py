import depthai as dai

pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.Camera)

print(type(cam))

print("\nAvailable methods:\n")

print(dir(cam))