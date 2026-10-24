"""
Step 2: detect a hand entering the candy bowl region.

Run:  python hand_in_bowl.py
Then: the first time, drag a box around the candy bowl and press ENTER.
      The box is saved to bowl_region.json and reused on later runs.

Keys in the video window:
  r       redraw the bowl region
  q/Esc   quit

Events are printed to the console:
  ENTER   a hand reached into the bowl region
  LINGER  the hand has stayed in the bowl for LINGER_SECONDS
  EXIT    the hand left the bowl (with how long it was in there)
  COUNT   how much candy the hand took, from candy_counter.py (needs OPENAI_API_KEY)
"""
import json
import os
import platform
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from candy_counter import count_candy, crop_to_hands

CAMERA_INDEX = 1       # Iriun phone camera, found with find_camera.py
LINGER_SECONDS = 3.0   # hand in the bowl this long counts as greedy
ENTER_FRAMES = 3       # frames in a row a hand must be inside before ENTER fires
EXIT_FRAMES = 5        # frames in a row with no hand inside before EXIT fires
BLACK_FRAME_MEAN = 5   # average brightness below this means the phone feed is dead

HERE = Path(__file__).parent
MODEL_PATH = HERE / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
REGION_PATH = HERE / "bowl_region.json"

# On Windows, the DirectShow backend opens cameras much faster than the default.
BACKEND = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY

GREEN, YELLOW, RED = (0, 200, 0), (0, 220, 255), (0, 0, 255)  # BGR


def ensure_model():
    """Download MediaPipe's hand model (~8 MB) on first run."""
    if not MODEL_PATH.exists():
        print(f"Downloading hand model to {MODEL_PATH.name} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def select_region(frame):
    """Let the user drag a box around the bowl. Returns (x1, y1, x2, y2) as 0-1 fractions."""
    window = "Drag a box around the candy bowl, then press ENTER"
    # Keep it in front: it often opens behind the terminal, and it only gets keys when focused.
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    x, y, w, h = cv2.selectROI(window, frame, showCrosshair=False)
    cv2.destroyWindow(window)
    if w == 0 or h == 0:
        return None
    fh, fw = frame.shape[:2]
    region = (x / fw, y / fh, (x + w) / fw, (y + h) / fh)
    REGION_PATH.write_text(json.dumps(region))
    return region


def load_region():
    if REGION_PATH.exists():
        return tuple(json.loads(REGION_PATH.read_text()))
    return None


def hand_in_region(landmarks, region):
    """True if any of the hand's 21 landmarks falls inside the bowl region."""
    x1, y1, x2, y2 = region
    return any(x1 <= lm.x <= x2 and y1 <= lm.y <= y2 for lm in landmarks)


class BowlWatcher:
    """Debounced hand-in-bowl state. update() returns an event name or None."""

    def __init__(self):
        self.inside = False
        self.streak = 0          # frames in a row that disagree with self.inside
        self.entered_at = 0.0
        self.lingered = False

    def update(self, hand_inside, now):
        if hand_inside == self.inside:
            self.streak = 0
        else:
            self.streak += 1
            if self.streak >= (ENTER_FRAMES if hand_inside else EXIT_FRAMES):
                self.inside = hand_inside
                self.streak = 0
                if hand_inside:
                    self.entered_at = now
                    self.lingered = False
                    return "ENTER"
                self.lingered = False
                return "EXIT"

        if self.inside and not self.lingered and self.dwell(now) >= LINGER_SECONDS:
            self.lingered = True
            return "LINGER"
        return None

    def dwell(self, now):
        return now - self.entered_at if self.inside else 0.0


def draw_overlay(frame, region, watcher, result, now, fps, count_label):
    fh, fw = frame.shape[:2]

    for landmarks in result.hand_landmarks:
        vision.drawing_utils.draw_landmarks(
            frame, landmarks, vision.HandLandmarksConnections.HAND_CONNECTIONS,
            vision.drawing_styles.get_default_hand_landmarks_style(),
            vision.drawing_styles.get_default_hand_connections_style())

    if watcher.lingered:
        color, label = RED, f"LINGERING {watcher.dwell(now):.1f}s"
    elif watcher.inside:
        color, label = YELLOW, f"HAND IN {watcher.dwell(now):.1f}s"
    else:
        color, label = GREEN, "BOWL: clear"

    x1, y1, x2, y2 = region
    p1, p2 = (int(x1 * fw), int(y1 * fh)), (int(x2 * fw), int(y2 * fh))
    cv2.rectangle(frame, p1, p2, color, 3)
    cv2.putText(frame, label, (p1[0], max(p1[1] - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"{fps:.0f} fps  hands: {len(result.hand_landmarks)}",
                (10, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"last grab: {count_label}", (10, fh - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if frame.mean() < BLACK_FRAME_MEAN:
        cv2.putText(frame, "BLACK FRAME - bring Iriun app to front on phone",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)


def main():
    ensure_model()

    cap = cv2.VideoCapture(CAMERA_INDEX, BACKEND)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {CAMERA_INDEX}. Is Iriun running?")

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    counting_enabled = bool(os.environ.get("OPENAI_API_KEY"))
    if not counting_enabled:
        print("OPENAI_API_KEY not set (see laptop/.env): candy counting is off, "
              "hand detection still works.")
    # API calls run on a worker thread so the video never freezes while we wait.
    executor = ThreadPoolExecutor(max_workers=1)
    pending = None             # (future, start time) of the count in flight
    count_label = "-" if counting_enabled else "counting off"
    last_hand = None           # (clean frame, hand landmarks) from the latest frame with a hand

    region = load_region()
    watcher = BowlWatcher()
    start = time.monotonic()
    last_ts = -1
    last_frame_time = start
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed, retrying...")
                time.sleep(0.2)
                continue

            if region is None:
                region = select_region(frame)
                if region is None:
                    raise SystemExit("No bowl region selected.")
                print(f"Bowl region saved to {REGION_PATH.name}")

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last_frame_time, 1e-6)
            last_frame_time = now

            # MediaPipe wants RGB and strictly increasing timestamps in VIDEO mode.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = max(int((now - start) * 1000), last_ts + 1)
            last_ts = ts
            result = landmarker.detect_for_video(mp_image, ts)

            # Remember the latest hand sighting. By the time EXIT fires, this is the
            # hand just outside the bowl, still holding whatever it grabbed.
            if result.hand_landmarks:
                last_hand = (frame.copy(), result.hand_landmarks)

            inside = any(hand_in_region(lms, region) for lms in result.hand_landmarks)
            dwell = watcher.dwell(now)  # grab before EXIT resets it
            event = watcher.update(inside, now)
            if event == "EXIT":
                print(f"[{now - start:7.1f}s] EXIT   hand was in bowl {dwell:.1f}s")
                if counting_enabled and last_hand and pending is None:
                    crop = crop_to_hands(*last_hand)
                    pending = (executor.submit(count_candy, crop), now)
                    count_label = "counting..."
            elif event:
                print(f"[{now - start:7.1f}s] {event}")

            if pending and pending[0].done():
                future, asked_at = pending
                pending = None
                try:
                    choice, confidence = future.result()
                    count_label = f"{choice} ({confidence:.2f})"
                    print(f"[{now - start:7.1f}s] COUNT  {choice} piece(s), "
                          f"confidence {confidence:.2f}, took {now - asked_at:.1f}s")
                except Exception as e:
                    count_label = "count failed"
                    print(f"[{now - start:7.1f}s] COUNT  failed: {e}")

            draw_overlay(frame, region, watcher, result, now, fps, count_label)
            cv2.imshow("Hand in bowl", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                region = select_region(frame) or region
                watcher = BowlWatcher()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
