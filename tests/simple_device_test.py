import depthai as dai

device = dai.Device()
print("Device connected successfully!")
print("MX ID:", device.getMxId())
device.close()