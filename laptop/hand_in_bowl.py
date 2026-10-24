"""
TreatOps: watches the candy box, counts what each trick-or-treater takes, and makes the plushie react.

Run:  python hand_in_bowl.py

The box is found by the ArUco markers on its corners (python box_tracker.py prints them), so it can be moved
while the camera stays put. Without markers, drag a box around it with r.

What happens:
  a hand comes near        eyes SUSPICIOUS, following the hand
  a hand reaches in        the costume roast starts being written in the background
  the hand comes out       the box before and after, and the hand, are sent to be counted
  1 piece                  eyes PLEASED: good choice
  2 pieces                 eyes SUSPICIOUS and a warning line
  3+ pieces, or a hand
  in the box for 3 s       the lid closes, eyes ANGRY, the roast, VICTIMS + 1; the lid opens after RESET_SECONDS
A trick-or-treater's total resets once no hand has been near the box for VISITOR_GONE_SECONDS.

The dashboard (dashboard.py) is at http://localhost:8080: the camera live, TOP BIG BACKS, roast replays and
buttons to drive it all.

Keys in the video window:
  r       draw the box by hand (and stop using the markers)
  a       go back to finding the box by its markers
  m       forget where the markers were (the box moved while some were covered)
  q/Esc   quit

Set SAVE_GRABS=1 in laptop/.env to keep each grab's photos in grabs/, to see what the counter was shown.
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
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import box_tracker
from box_tracker import BoxTracker
from candy_counter import PIECES, changed_fraction, count_taken, crop_to_hands, sharpness
from dashboard import Dashboard
from plushie import Plushie, wav_seconds
from roaster import Roaster

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX") or 1)  # Iriun phone camera, found with find_camera.py
FRAME_SIZE = (1280, 720)    # ask for 720p: the default 640x480 is too coarse to tell wrappers apart
LINGER_SECONDS = 3.0        # hand in the box this long counts as greedy
ENTER_FRAMES = 3            # frames in a row a hand must be inside before ENTER fires
EXIT_FRAMES = 8             # frames in a row with no hand inside before EXIT fires (tracking drops out
                            # for a few frames when the box's walls hide the hand)
SETTLE_FRAMES = 8           # frames after EXIT before the "after" photo (and the hand's sharpest look)...
SETTLE_MAX_FRAMES = 30      # ...once no hand is near the box, or after this many frames regardless
CLEAR_FRAMES = 10           # frames with no hand near the box before the "before" photo is refreshed
NEAR = 1.6                  # "near the box": the box outline grown this much
MARKER_EVERY = 3            # look for the markers every this many frames
STUCK_SECONDS = 20.0        # a "hand" in the box this long is something else (a prop): forget it
VISITOR_GONE_SECONDS = 5.0  # no hand near the box this long: the next hand is a new trick-or-treater
RESET_SECONDS = 3.0         # after a scare, when the lid opens and the eyes calm down
ROAST_WAIT_SECONDS = 2.0    # how long a scare waits for the costume roast before using a canned one
LOOK_EVERY_SECONDS = 0.25   # how often the eyes are told where the hand is
LOOK_FLIP = os.environ.get("LOOK_FLIP") == "1"  # set if the eyes look away from the hand instead of at it
BLACK_FRAME_MEAN = 5        # average brightness below this means the phone feed is dead
SAVE_GRABS = os.environ.get("SAVE_GRABS") == "1"

HERE = Path(__file__).parent
MODEL_PATH = HERE / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
REGION_PATH = HERE / "bowl_region.json"
GRABS = HERE / "grabs"

# On Windows, the DirectShow backend opens cameras much faster than the default.
BACKEND = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY

GREEN, YELLOW, RED, WHITE = (0, 200, 0), (0, 220, 255), (0, 0, 255), (255, 255, 255)  # BGR


def ensure_model():
    """Download MediaPipe's hand model (~8 MB) on first run."""
    if not MODEL_PATH.exists():
        print(f"Downloading hand model to {MODEL_PATH.name} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def rectangle(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


def select_region(frame):
    """Let the user drag a box around the candy box. Returns its outline, or None."""
    window = "Drag a box around the candy box, then press ENTER"
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
    return rectangle(*region)


def load_region():
    if REGION_PATH.exists():
        return rectangle(*json.loads(REGION_PATH.read_text()))
    return None


def ordinal(n):
    """1st, 2nd, 3rd, 4th... 11th, 12th, 13th... 21st."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def hand_in(landmarks, outline):
    """True if any of the hand's 21 landmarks falls inside the outline."""
    return any(box_tracker.contains(outline, lm.x, lm.y) for lm in landmarks)


class BowlWatcher:
    """Debounced hand-in-box state. update() returns an event name or None."""

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


class Show:
    """
    What the plushie does about each trick-or-treater: their running total, the tiers and the scare. Each one
    goes on the dashboard's leaderboard, with the photo their roast was written from.
    """

    def __init__(self, plushie, roaster, dashboard):
        self.plushie = plushie
        self.roaster = roaster
        self.dashboard = dashboard
        self.victims = 0
        self.visitor = None    # this trick-or-treater: {"id", "taken", "roast", "warned", "scared"}
        self.watching = False  # a hand is (or was just) near the box: eyes suspicious
        self.last_hand = 0.0
        self.reopen_at = None
        plushie.lid(closed=False)
        plushie.eyes("CALM")
        plushie.count(0)

    def _arrive(self, frame):
        """A new trick-or-treater: their roast starts being written from this frame of their costume."""
        visitor_id = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        self.visitor = {"id": visitor_id, "taken": 0, "warned": False, "scared": False,
                        "roast": self.roaster.prepare(frame) if frame is not None else None}
        self.dashboard.visitor_started(visitor_id, frame)
        self.dashboard.log("Someone's at the box")
        print("  new trick-or-treater")

    def reach(self, frame):
        """A hand went into the box. The first time for this trick-or-treater, start their roast."""
        if self.visitor is None:
            self._arrive(frame)

    def took(self, pieces, now):
        """A grab was counted."""
        if self.visitor is None:  # counted after they'd gone: still theirs, but no costume to roast
            self._arrive(None)
        visitor = self.visitor
        visitor["taken"] += pieces
        print(f"  this trick-or-treater has taken {visitor['taken']}")
        self.dashboard.visitor_took(visitor["id"], visitor["taken"])
        self.dashboard.log(f"Took {pieces} · {visitor['taken']} this visit")
        if visitor["scared"]:
            return  # already caught: the lid's shut and the roast is playing
        if visitor["taken"] >= 3:
            self.scare(now)
        elif visitor["taken"] == 2 and not visitor["warned"]:
            visitor["warned"] = True
            self.plushie.eyes("SUSPICIOUS")
            self.roaster.say_warning()
        elif visitor["taken"] == 1:
            self.plushie.eyes("PLEASED")  # good choice

    def scare(self, now):
        visitor = self.visitor
        if visitor is None or visitor["scared"]:
            return
        visitor["scared"] = True
        self.victims += 1
        print(f"  SCARE! victims: {self.victims}")
        self.plushie.lid(closed=True)
        self.plushie.eyes("ANGRY")
        self.plushie.count(self.victims)
        self.dashboard.caught(visitor["id"], now)
        self.dashboard.log(f"Caught · {ordinal(self.victims)} tonight")
        visitor_id = visitor["id"]
        self.roaster.roast(visitor["roast"], wait=ROAST_WAIT_SECONDS,
                           on_spoken=lambda text, mp3: self.dashboard.roast_spoken(visitor_id, text, mp3))
        self.reopen_at = now + RESET_SECONDS

    def force_scare(self, frame, now):
        """The dashboard's scare button: whoever's there gets it (a new visitor if nobody is)."""
        if self.visitor is None:
            self._arrive(frame)
        self.scare(now)

    def leave(self):
        """The current trick-or-treater is done (they left, or the dashboard said so)."""
        if self.visitor is not None:
            print(f"  trick-or-treater gone, took {self.visitor['taken']}")
            self.dashboard.log(f"Left with {self.visitor['taken']}")
        self.visitor = None
        self.watching = False
        if self.reopen_at is None:
            self.plushie.eyes("CALM")

    def tick(self, now, hand_near, busy):
        """Every frame: eyes on approach, the lid back open after a scare, the visitor leaving."""
        if hand_near:
            self.last_hand = now
            if not self.watching and self.reopen_at is None:
                self.watching = True
                self.plushie.eyes("SUSPICIOUS")  # I see you
        if self.reopen_at is not None and now >= self.reopen_at:
            self.reopen_at = None
            self.plushie.lid(closed=False)
            self.plushie.eyes("CALM")
        gone = now - self.last_hand > VISITOR_GONE_SECONDS
        if gone and not busy and (self.visitor is not None or self.watching):
            self.leave()


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, BACKEND)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {CAMERA_INDEX}. Is Iriun running?")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    print(f"Camera {CAMERA_INDEX}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap


def save_grab(before, after, hand):
    GRABS.mkdir(exist_ok=True)
    stamp = time.strftime("%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    for name, image in (("before", before), ("after", after), ("hand", hand)):
        if image is not None and image.size:
            cv2.imwrite(str(GRABS / f"{stamp}-{name}.jpg"), image)


def draw_overlay(frame, outline, source, watcher, result, now, fps, status):
    fh, fw = frame.shape[:2]

    for landmarks in result.hand_landmarks:
        vision.drawing_utils.draw_landmarks(
            frame, landmarks, vision.HandLandmarksConnections.HAND_CONNECTIONS,
            vision.drawing_styles.get_default_hand_landmarks_style(),
            vision.drawing_styles.get_default_hand_connections_style())

    if outline is None:
        cv2.putText(frame, "Show the box's markers, or press r to draw it", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)
    else:
        if watcher.lingered:
            color, label = RED, f"LINGERING {watcher.dwell(now):.1f}s"
        elif watcher.inside:
            color, label = YELLOW, f"HAND IN {watcher.dwell(now):.1f}s"
        else:
            color, label = GREEN, "BOX: clear"
        box_tracker.draw(frame, outline, color)
        x, y = (outline.min(axis=0) * (fw, fh)).astype(int)
        cv2.putText(frame, f"{label}  ({source})", (x, max(y - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, f"{fps:.0f} fps  hands: {len(result.hand_landmarks)}",
                (10, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    cv2.putText(frame, status, (10, fh - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

    if frame.mean() < BLACK_FRAME_MEAN:
        cv2.putText(frame, "BLACK FRAME - bring Iriun app to front on phone",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)


def main():
    ensure_model()
    cap = open_camera()

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    dashboard = Dashboard()
    plushie = Plushie(os.environ.get("PLUSHIE_HOST"))
    roaster = Roaster(plushie)
    show = Show(plushie, roaster, dashboard)

    counting_enabled = bool(os.environ.get("OPENAI_API_KEY"))
    if not counting_enabled:
        print("OPENAI_API_KEY not set (see laptop/.env): candy counting is off, "
              "hand detection still works.")
    # API calls run on worker threads so the video never freezes while we wait.
    executor = ThreadPoolExecutor(max_workers=2)
    counts = []                # grabs being counted: (future, asked at)
    last_count = "-" if counting_enabled else "counting off"

    tracker = BoxTracker()
    manual = load_region()     # a box drawn by hand, used when there are no markers
    use_markers = True
    watcher = BowlWatcher()
    before = None              # the box with no hand near it: the "before" photo of the next grab
    clear = 0                  # frames in a row with no hand near the box
    grab = None                # the grab being finished: {"before", "since_exit", "hand": (sharpness, crop)}

    start = time.monotonic()
    last_ts = -1
    last_frame_time = start
    last_look = 0.0
    fps = 0.0
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed, retrying...")
                if cv2.waitKey(200) & 0xFF in (ord("q"), 27):  # keep the window answering meanwhile
                    break
                continue
            frame_index += 1

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last_frame_time, 1e-6)
            last_frame_time = now
            dashboard.feed(frame, now)  # the clean picture, for replays

            # The dashboard's buttons.
            for action, value in dashboard.controls():
                shown = f"{len(value)} bytes" if isinstance(value, bytes) else ("" if value is None else value)
                print(f"[{now - start:7.1f}s] DASHBOARD {action} {shown}")
                if action == "say":  # the operator talking through the dashboard's mic
                    roaster.say_recording(value)
                    dashboard.log(f"Operator on the speaker ({wav_seconds(value):.1f}s)")
                elif action == "scare":
                    show.force_scare(frame, now)
                elif action == "roast":  # roast whoever's in front of the camera, right now
                    roaster.roast(roaster.prepare(frame), wait=12.0,
                                  on_spoken=lambda text, mp3: dashboard.log(f'Roasted: "{text}"'))
                elif action == "end_visitor":
                    show.leave()
                elif action == "lid":
                    plushie.lid(closed=bool(value))
                elif action == "eyes":
                    plushie.eyes(value)
                elif action == "mute":
                    roaster.muted = bool(value)
                    dashboard.log("Voice muted" if roaster.muted else "Voice on")
                elif action == "clear_board":
                    dashboard.clear()

            # Where the box is: its markers (looked for every few frames), else the hand-drawn box.
            if use_markers and frame_index % MARKER_EVERY == 1:
                tracker.update(frame)
            outline = tracker.outline() if use_markers else None
            source = f"markers {tracker.visible}/4" if outline is not None else "drawn"
            if outline is None:
                outline = manual

            # MediaPipe wants RGB and strictly increasing timestamps in VIDEO mode.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = max(int((now - start) * 1000), last_ts + 1)
            last_ts = ts
            result = landmarker.detect_for_video(mp_image, ts)
            hands = result.hand_landmarks

            near = False
            if outline is not None:
                inside = any(hand_in(lms, outline) for lms in hands)
                near_outline = box_tracker.grow(outline, NEAR)
                near_hands = [lms for lms in hands if hand_in(lms, near_outline)]
                near = bool(near_hands)
                event = watcher.update(inside, now)

                # The eyes follow the hand nearest the box (a few times a second; the plushie is on Wi-Fi).
                if near_hands and now - last_look >= LOOK_EVERY_SECONDS:
                    last_look = now
                    x = sum(lm.x for lm in near_hands[0]) / len(near_hands[0])
                    y = sum(lm.y for lm in near_hands[0]) / len(near_hands[0])
                    # The camera faces the way the plushie does: a hand on the camera's left is on the
                    # plushie's left, which is the face's right as you look at it.
                    plushie.look((x - 0.5) * 2 if LOOK_FLIP else (0.5 - x) * 2, (y - 0.5) * 2)

                # Something that never leaves the box isn't a hand reaching in (a prop, a glove decoration).
                if watcher.inside and watcher.dwell(now) > STUCK_SECONDS:
                    print(f"[{now - start:7.1f}s] a 'hand' has been in the box {STUCK_SECONDS:.0f}s: ignoring it")
                    watcher, grab = BowlWatcher(), None

                if event == "ENTER":
                    print(f"[{now - start:7.1f}s] ENTER")
                    if grab is None:
                        grab = {"before": before, "since_exit": None, "hand": None}
                    else:
                        grab["since_exit"] = None  # back in before the last grab was finished: same grab
                    show.reach(frame)
                elif event == "LINGER":
                    print(f"[{now - start:7.1f}s] LINGER")
                    show.scare(now)
                elif event == "EXIT":
                    print(f"[{now - start:7.1f}s] EXIT")
                    if grab is not None:
                        grab["since_exit"] = 0

                # Finishing a grab: the sharpest look at the hand on its way out, then the box once it's
                # settled, and the three photos go off to be counted.
                if grab is not None and grab["since_exit"] is not None and not watcher.inside:
                    grab["since_exit"] += 1
                    leaving = [lms for lms in near_hands if not hand_in(lms, outline)]
                    if leaving:
                        crop = crop_to_hands(frame, leaving)
                        score = sharpness(crop)
                        if grab["hand"] is None or score > grab["hand"][0]:
                            grab["hand"] = (score, crop.copy())
                    settled = grab["since_exit"] >= SETTLE_FRAMES and not near
                    if settled or grab["since_exit"] >= SETTLE_MAX_FRAMES:
                        after = box_tracker.crop(frame, outline)
                        hand = grab["hand"][1] if grab["hand"] else None
                        changed = changed_fraction(grab["before"], after)
                        print(f"[{now - start:7.1f}s] GRAB   box changed {changed:.1%}, "
                              f"{'hand seen' if hand is not None else 'hand not seen'}")
                        if SAVE_GRABS:
                            save_grab(grab["before"], after, hand)
                        if counting_enabled:
                            counts.append((executor.submit(count_taken, grab["before"], after, hand), now))
                            last_count = "counting..."
                        before = after  # the next grab starts from the box as this one left it
                        grab = None
                        clear = 0

                # The "before" photo, kept fresh while nobody's near the box.
                if grab is None and not watcher.inside:
                    clear = 0 if near else clear + 1
                    if clear >= CLEAR_FRAMES:
                        before = box_tracker.crop(frame, outline)
                        clear = 0

            for future, asked_at in [c for c in counts if c[0].done()]:
                counts.remove((future, asked_at))
                try:
                    choice, confidence = future.result()
                    confidence = float(confidence or 0)
                except Exception as e:
                    last_count = "count failed"
                    print(f"[{now - start:7.1f}s] COUNT  failed: {e}")
                    dashboard.log("Couldn't count that grab")
                    continue
                last_count = f"{choice} ({confidence:.2f})"
                print(f"[{now - start:7.1f}s] COUNT  {choice} piece(s), confidence {confidence:.2f}, "
                      f"took {now - asked_at:.1f}s")
                pieces = PIECES.get(choice)  # "unclear" and "refused" don't count either way
                if pieces is not None:
                    show.took(pieces, now)

            show.tick(now, hand_near=near, busy=grab is not None or bool(counts))

            visitor = show.visitor
            status = (f"last grab: {last_count}   this visitor: {visitor['taken'] if visitor else '-'}"
                      f"   victims: {show.victims}")
            dashboard.set_status(
                victims=show.victims, visitor=visitor["taken"] if visitor else None,
                warned=bool(visitor and visitor["warned"]), scared=bool(visitor and visitor["scared"]),
                last_grab=last_count.split(" (")[0],  # the count without the model's confidence
                hand_in=watcher.inside, hand_in_for=round(watcher.dwell(now), 1), linger=LINGER_SECONDS,
                speaking=roaster.speaking, fps=round(fps), box=source if outline is not None else None,
                muted=roaster.muted, plushie=plushie.host, eyes=plushie.mood, lid_closed=plushie.lid_closed,
                counting=counting_enabled)
            draw_overlay(frame, outline, source, watcher, result, now, fps, status)
            dashboard.show(frame, now)
            cv2.imshow("TreatOps", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                drawn = select_region(frame)
                if drawn is not None:
                    manual = drawn
                    use_markers = False
                watcher, grab, before, clear = BowlWatcher(), None, None, 0
            elif key == ord("a"):
                use_markers = True
                watcher, grab, before, clear = BowlWatcher(), None, None, 0
            elif key == ord("m"):
                tracker.forget()
                watcher, grab, before, clear = BowlWatcher(), None, None, 0
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        roaster.close()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
