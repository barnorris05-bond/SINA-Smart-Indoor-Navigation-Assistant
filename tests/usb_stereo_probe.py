"""
tests/usb_stereo_probe.py

Read-only OAK-D Lite USB + camera-hardware probe (Phase 2 debugging).

Answers, without building any pipeline and without touching SINA code:
  1. Is a Luxonis device visible to DepthAI at all?
  2. What USB speed was negotiated?  (UsbSpeed.SUPER = USB 3.x, HIGH = USB 2.0)
  3. Which camera sockets does the Myriad X physically detect?
     (USB-speed independent - isolates the "empty stereo pairs" issue)
  4. Which stereo pairs are available?

Run:  ./.venv/Scripts/python.exe tests/usb_stereo_probe.py

Exit codes:
  0 = probe completed (interpret the printed report)
  1 = device enumeration/open failed (e.g. wedged by an earlier crash)
  2 = no device visible (unplugged / cable / port)
"""

from __future__ import annotations

import sys

import depthai as dai


def print_device_info(info: dai.DeviceInfo) -> None:
    print(f"  {info}")
    for attr in ("name", "mxid", "deviceId", "state", "usbSpeed", "speed", "platform"):
        if hasattr(info, attr):
            print(f"    {attr} = {getattr(info, attr)}")


def main() -> int:
    print(f"depthai version: {dai.__version__}")

    # --------------------------------------------------------------
    # 1. Enumeration WITHOUT booting the device
    # --------------------------------------------------------------
    try:
        infos = dai.Device.getAllConnectedDevices()
    except Exception as exc:
        print(f"ENUMERATION ERROR: {type(exc).__name__}: {exc}")
        return 1

    if not infos:
        print("NO DEVICE visible to DepthAI.")
        print("  -> Reconnect the OAK-D directly to a laptop USB 3 port (no hub), then rerun.")
        return 2

    print(f"\nVisible devices: {len(infos)}")
    for info in infos:
        print_device_info(info)

    # --------------------------------------------------------------
    # 2. Boot the device (read-only queries only - no pipeline)
    # --------------------------------------------------------------
    try:
        device = dai.Device()
    except Exception as exc:
        print(f"\nDEVICE OPEN FAILED: {type(exc).__name__}: {exc}")
        print("  -> If a previous run crashed, unplug/replug the OAK-D and rerun.")
        return 1

    try:
        # USB speed as negotiated by the current link.
        try:
            print(f"\nUSB speed: {device.getUsbSpeed()}")
        except Exception as exc:
            print(f"\nUSB speed: error {type(exc).__name__}: {exc}")

        for name in ("getPlatform", "getPlatformAsString", "getProductName", "getDeviceName"):
            if hasattr(device, name):
                try:
                    print(f"{name}: {getattr(device, name)()}")
                except Exception as exc:
                    print(f"{name}: error {type(exc).__name__}: {exc}")

        # ----------------------------------------------------------
        # 3. Camera hardware detection (USB-speed independent)
        # ----------------------------------------------------------
        print("\nCamera detection:")
        enumerated = False
        for name in ("getCameraSockets", "getAvailableCameras", "getCameraFeatures"):
            if hasattr(device, name):
                try:
                    result = getattr(device, name)()
                    print(f"  {name} -> {result}")
                    enumerated = True
                except Exception as exc:
                    print(f"  {name} error: {type(exc).__name__}: {exc}")

        if hasattr(device, "getAvailableStereoPairs"):
            try:
                pairs = device.getAvailableStereoPairs()
                print(f"  getAvailableStereoPairs -> {pairs}")
            except Exception as exc:
                print(f"  getAvailableStereoPairs error: {type(exc).__name__}: {exc}")

        if not enumerated:
            # Fall back to raw discovery: list every plausibly relevant member.
            relevant = [
                m
                for m in dir(device)
                if any(k in m.lower() for k in ("camera", "stereo", "socket"))
            ]
            print("  No known enumeration method produced a result; relevant members:")
            for member in relevant:
                print(f"    {member}")
    finally:
        if hasattr(device, "close"):
            try:
                device.close()
            except Exception:
                pass

    print("\nProbe finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
