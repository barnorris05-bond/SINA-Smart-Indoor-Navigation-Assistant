import cv2

from camera.camera_manager import CameraManager


camera = CameraManager()

camera.start()

print("Press Q to quit.")

try:

    while True:

        frame = camera.get_rgb_frame()

        if frame is None:
            continue

        cv2.imshow("OAK-D Lite RGB", frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

finally:

    camera.stop()

    cv2.destroyAllWindows()