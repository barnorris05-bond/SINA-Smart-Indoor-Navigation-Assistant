"""
tests/inspect_device.py
Inspect how Device should be used in DepthAI v3.7.1
"""

import depthai as dai
import inspect

print("=" * 60)
print("DepthAI Device API Inspection")
print("=" * 60)

print("DepthAI Version:", dai.__version__)
print()

# Inspect Device class
print("dir(dai.Device):")
print([x for x in dir(dai.Device) if not x.startswith('_')])

print("\n" + "=" * 60)
print("Constructor Signature:")
print("=" * 60)

try:
    print(inspect.signature(dai.Device.__init__))
except Exception as e:
    print("Signature inspection failed:", e)
    print("\nUsing help() instead:")
    help(dai.Device)