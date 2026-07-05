"""
tests/test_rgb_pipeline.py
Simple test to verify CameraManager is working with FPS display.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from camera.camera_manager import CameraManager
import cv2
from utils.fps import FPS   # FPS counter


def main():
    camera = CameraManager()
    fps_counter = FPS()        # Initialize FPS counter
    
    print("Starting camera...")
    camera.start()
    
    print("Camera started. Press 'q' to quit.")
    
    try:
        for i in range(300):   # Increased to 300 frames for better testing
            frame = camera.get_rgb_frame()
            
            if frame is not None:
                # Update FPS and draw it on the frame
                current_fps = fps_counter.update()
                
                cv2.putText(
                    frame,
                    f"FPS: {current_fps}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow("RGB Preview - Smart Navigation Assistant", frame)
                print(f"Frame {i+1} received: {frame.shape} | FPS: {current_fps}")
            else:
                print("No frame received")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()