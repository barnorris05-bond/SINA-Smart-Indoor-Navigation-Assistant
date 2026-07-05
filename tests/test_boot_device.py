import depthai as dai

print("=" * 50)
print("Booting OAK-D Lite...")
print("=" * 50)

try:
    with dai.Device() as device:
        print("✅ Device booted successfully!")
        print(f"Connected Device: {device}")

except Exception as e:
    print("❌ Failed to boot device.")
    print(e)