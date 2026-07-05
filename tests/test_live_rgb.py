"""
test_live_rgb.py

Display a live RGB stream from the OAK-D Lite.
"""

import cv2

from camera.camera_manager import CameraManager


def main():

    camera = CameraManager()

    print("Starting camera...")

    camera.start()

    print("Press 'Q' to quit.\n")

    try:

        while True:

            frame = camera.get_rgb_frame()

            if frame is None:
                continue

            cv2.imshow("OAK-D Lite RGB", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:

        print("Stopping camera...")

        camera.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()