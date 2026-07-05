"""
tests/test_rgb_pipeline.py
Simple test to verify CameraManager is working.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from camera.camera_manager import CameraManager
import cv2

def main():
    camera = CameraManager()
    
    print("Starting camera...")
    camera.start()
    
    print("Camera started. Press 'q' to quit.")
    
    try:
        for i in range(100):  # Try to get 100 frames
            frame = camera.get_rgb_frame()
            if frame is not None:
                cv2.imshow("RGB Preview", frame)
                print(f"Frame {i+1} received: {frame.shape}")
            else:
                print("No frame received")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()