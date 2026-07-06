import depthai as dai

print("Trying to connect...")

devices = dai.Device.getAllAvailableDevices()
if not devices:
    print("No devices found")
else:
    d = devices[0]
    print("Using device:", d)
    try:
        device = dai.Device(d)
        print("SUCCESS! Device connected.")
        print("MX ID:", device.getMxId())
        device.close()
    except Exception as e:
        print("Failed to connect:", e)