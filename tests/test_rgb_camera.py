import cv2
import depthai as dai

print("Creating pipeline...")

# Create pipeline
pipeline = dai.Pipeline()

# Create RGB camera node
cam = pipeline.create(dai.node.ColorCamera)

cam.setPreviewSize(640, 480)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Create output stream
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("rgb")

cam.preview.link(xout.input)

print("Connecting to OAK-D Lite...")

# Connect to device
with dai.Device(pipeline) as device:

    print("Camera connected successfully!")
    print("Press Q to quit.")

    q_rgb = device.getOutputQueue(
        name="rgb",
        maxSize=4,
        blocking=False
    )

    while True:

        frame = q_rgb.get().getCvFrame()

        cv2.imshow("OAK-D Lite RGB Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()