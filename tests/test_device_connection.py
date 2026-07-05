import depthai as dai

print("=" * 50)
print("Searching for OAK devices...")
print("=" * 50)

devices = dai.Device.getAllAvailableDevices()

if not devices:
    print("❌ No OAK devices found.")
    exit()

print(f"✅ Found {len(devices)} device(s).\n")

for index, device in enumerate(devices, start=1):
    print(f"Device {index}")
    print(f"Name      : {device.name}")
    print(f"Device ID : {device.deviceId}")
    print(f"State     : {device.state}")
    print(f"Protocol  : {device.protocol}")
    print("-" * 50)