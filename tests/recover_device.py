import depthai as dai
import time

print("Attempting device recovery...")

for i in range(5):
    try:
        devices = dai.Device.getAllAvailableDevices()
        if devices:
            d = devices[0]
            print(f"Found device: {d}")
            device = dai.Device(d)
            print("Device connected. Closing to reset...")
            device.close()
            time.sleep(2)
            print("Recovery attempt", i+1, "complete.")
        else:
            print("No device found.")
    except Exception as e:
        print("Error:", e)
    time.sleep(1)

print("Recovery finished. Try the stereo test now.")