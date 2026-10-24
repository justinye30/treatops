"""
Step 1: find which camera index is the Iriun (phone) webcam.

Run:  python find_camera.py
Then: a window opens for each camera that works. Look at it, press any key to
      move to the next one. Note the index that shows your phone's view.
"""
import platform
import cv2

# On Windows, the DirectShow backend opens cameras much faster than the default.
BACKEND = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY

working = []

for index in range(5):  # most laptops have < 5 cameras; Iriun is usually 1 or 2
    cap = cv2.VideoCapture(index, BACKEND)
    if not cap.isOpened():
        continue

    ok, frame = cap.read()
    if not ok:
        cap.release()
        continue

    h, w = frame.shape[:2]
    print(f"Camera {index}: working, {w}x{h}")
    working.append(index)

    cv2.putText(frame, f"Camera {index} - press any key", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera test", frame)
    cv2.waitKey(0)  # wait until you press a key
    cap.release()

cv2.destroyAllWindows()
print(f"\nWorking cameras: {working}")
print("Use the index that showed your phone's view as CAMERA_INDEX in the next script.")
