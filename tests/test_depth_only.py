""" tests/test_depth_only.py

Minimal stereo depth hardware test for DepthAI v3.
"""

import sys
import os
import cv2
import depthai as dai

sys.path.insert(0, os.path.abspath("."))
from camera.stereo_pipeline import StereoPipelineBuilder

def main():
    frame_count = 0
    print("Creating minimal depth pipeline...")
    pipeline, depth_out = StereoPipelineBuilder.create()
    
    print("Starting device...")
    try:
        device = dai.Device()
        device.startPipeline(pipeline)
        
        print("Device started. Creating queue...")
        depth_queue = depth_out.createOutputQueue(maxSize=4, blocking=False)
        print("Streaming depth. Press 'q' to quit.")
        
        while True:
            frame = depth_queue.tryGet()
            if frame is not None:
                depth_data = frame.getFrame()
                frame_count += 1
                print(
                    f"\rDepth frames received: {frame_count} | "
                    f"Shape: {depth_data.shape} | "
                    f"Dtype: {depth_data.dtype}", 
                    end=""
                )
                
                depth_norm = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                cv2.imshow("Depth Only", depth_colored)
                
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
    except Exception as e:
        print("\n\nDepth test failed:")
        print(type(e).__name__, str(e))
    finally:
        try:
            device.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print(f"\nDepth frames received: {frame_count}")
        print("Depth test finished.")

if __name__ == "__main__":
    main()
