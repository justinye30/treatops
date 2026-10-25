"""
TreatOps: watches the candy box, counts what each trick-or-treater takes, and makes the plushie react.

Run:  python hand_in_bowl.py

It opens the dashboard in the browser; that's the whole interface (Ctrl+C here stops it). The box is found
by the ArUco markers on its corners (python box_tracker.py prints them), so it can be moved while the camera
stays put. Without markers, press "Draw box" on the dashboard and drag over the candy box on the camera.

What happens:
  a hand shows up          a welcome: take some candy, but not too much
  a hand comes near        eyes SUSPICIOUS, following the hand
  a hand reaches in        the costume roast starts being written in the background
  the hand comes out       the box before and after, and the hand, are counted at once
  1 piece                  eyes PLEASED: good choice
  2 pieces                 eyes SUSPICIOUS and a warning line
  3+ pieces                the lid closes, eyes ANGRY, the roast, VICTIMS + 1; the lid opens once the roast is over
A trick-or-treater's total resets once no hand has been near the box for VISITOR_GONE_SECONDS.

The dashboard (dashboard.py) is at http://localhost:8080: the camera live with the box and hands drawn on
it, TOP BIG BACKS (two pieces or more), roast replays and buttons to drive it all.
Set VIDEO_WINDOW=1 in laptop/.env to also get the OpenCV window on this laptop (q/Esc in it quits).

Set SAVE_GRABS=1 in laptop/.env to keep each grab's photos in grabs/, to see what the counter was shown.
"""
import json
import os
import platform
import time
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import box_tracker
from box_finder import find_box
from box_tracker import BoxTracker
from candy_counter import PIECES, changed_fraction, count_taken, crop_to_hands, sharpness
from dashboard import PORT as DASHBOARD_PORT, Dashboard
from plushie import Plushie, wav_seconds
from roaster import Roaster

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX") or 1)  # Iriun phone camera, found with find_camera.py
FRAME_SIZE = (1280, 720)    # ask for 720p: the default 640x480 is too coarse to tell wrappers apart
# Catches come from the count, decided as the hand leaves the box. A hand left in the box this long can be
# caught too, but that's off (0) by default: picking out one candy can take a while.
LINGER_SECONDS = float(os.environ.get("LINGER_SECONDS") or 0)
ENTER_FRAMES = 3            # frames in a row a hand must be inside before ENTER fires
EXIT_FRAMES = 5             # frames in a row with no hand inside before EXIT fires (tracking drops out
                            # for a frame or two when the box's walls hide the hand)
SETTLE_FRAMES = 3           # frames after EXIT before the "after" photo (and the hand's sharpest look)...
SETTLE_MAX_FRAMES = 12      # ...once no hand is near the box, or after this many frames regardless
CLEAR_FRAMES = 10           # frames with no hand near the box before the "before" photo is refreshed
NEAR = 1.6                  # "near the box": the box outline grown this much
MARKER_EVERY = 3            # look for the markers every this many frames
STUCK_SECONDS = 20.0        # a "hand" in the box this long is something else (a prop): forget it
VISITOR_GONE_SECONDS = 5.0  # no hand near the box this long: the next hand is a new trick-or-treater
REFIND_SECONDS = 20.0       # without markers, look for the box again this often while nobody's at it
NO_BOX_RETRY_SECONDS = 5.0  # ...and this soon when it hasn't been found at all yet
BOX_MOVED_IOU = 0.8         # a box found that overlaps the current one less than this has moved: use it
MAX_LID_SECONDS = 30.0     # the lid opens when the roast is over, or after this long if the roast never comes
ROAST_WAIT_SECONDS = 2.0    # how long a scare waits for the costume roast before using a canned one
LOOK_EVERY_SECONDS = 0.25   # how often the eyes are told where the hand is
LOOK_FLIP = os.environ.get("LOOK_FLIP") == "1"  # set if the eyes look away from the hand instead of at it
BLACK_FRAME_MEAN = 5        # average brightness below this means the phone feed is dead
SAVE_GRABS = os.environ.get("SAVE_GRABS") == "1"
# For a demo: every grab counts as this many pieces, without asking the counter (so 3 means every grab is a
# catch, at once). Empty or 0 counts for real.
DEMO_PIECES = int(os.environ.get("DEMO_PIECES") or 0)
VIDEO_WINDOW = os.environ.get("VIDEO_WINDOW") == "1"  # the OpenCV window too; the dashboard is the interface

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


def save_region(region):
    """A box drawn on the dashboard, (x1, y1, x2, y2) as 0-1 fractions: kept for next time. Returns its outline."""
    REGION_PATH.write_text(json.dumps(region))
    return rectangle(*region)


def hide_box(frame, outline):
    """The frame with the candy box painted over, for the roast: the box isn't a costume, the person is."""
    if frame is None or outline is None:
        return frame
    fh, fw = frame.shape[:2]
    hidden = frame.copy()
    cv2.fillPoly(hidden, [(box_tracker.grow(outline, 1.05) * (fw, fh)).astype(np.int32)], (90, 90, 90))
    return hidden


def overlap(a, b):
    """How much two (x1, y1, x2, y2) boxes overlap: intersection over union, 0-1."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


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

        if LINGER_SECONDS and self.inside and not self.lingered and self.dwell(now) >= LINGER_SECONDS:
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
        self.welcomed = False  # this trick-or-treater has heard the welcome
        self.last_hand = 0.0
        self.reopen_at = None   # during a scare: the latest the lid stays shut
        self.roast_done = None  # during a scare: the Event the roaster sets once the roast's been said
        plushie.lid(closed=False)
        plushie.eyes("CALM")
        plushie.count(0)

    def _arrive(self, frame, outline=None):
        """A new trick-or-treater: their roast starts being written from this frame of their costume."""
        visitor_id = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        self.visitor = {"id": visitor_id, "taken": 0, "warned": False, "scared": False,
                        "roast": self.roaster.prepare(hide_box(frame, outline)) if frame is not None else None}
        self.dashboard.visitor_started(visitor_id, frame)
        self.dashboard.log("Someone's at the box")
        print("  new trick-or-treater")

    def reach(self, frame, outline):
        """A hand went into the box. The first time for this trick-or-treater, start their roast."""
        if self.visitor is None:
            self._arrive(frame, outline)

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
            self.scare(now, f"took {visitor['taken']}")
        elif visitor["taken"] == 2 and not visitor["warned"]:
            visitor["warned"] = True
            self.plushie.eyes("SUSPICIOUS")
            self.roaster.say_warning()
        elif visitor["taken"] == 1:
            self.plushie.eyes("PLEASED")  # good choice

    def scare(self, now, why):
        visitor = self.visitor
        if visitor is None or visitor["scared"]:
            return
        visitor["scared"] = True
        self.victims += 1
        print(f"  SCARE ({why})! victims: {self.victims}")
        self.plushie.lid(closed=True)
        self.plushie.eyes("ANGRY")
        self.plushie.count(self.victims)
        self.dashboard.caught(visitor["id"], now)
        self.dashboard.log(f"Caught · {ordinal(self.victims)} tonight · {why}")
        visitor_id = visitor["id"]
        self.roast_done = self.roaster.roast(
            visitor["roast"], wait=ROAST_WAIT_SECONDS,
            on_spoken=lambda text, mp3: self.dashboard.roast_spoken(visitor_id, text, mp3))
        self.reopen_at = now + MAX_LID_SECONDS

    def force_scare(self, frame, now, outline):
        """The dashboard's scare button: whoever's there gets it (a new visitor if nobody is)."""
        if self.visitor is None:
            self._arrive(frame, outline)
        self.scare(now, "scare button")

    def leave(self):
        """The current trick-or-treater is done (they left, or the dashboard said so)."""
        if self.visitor is not None:
            print(f"  trick-or-treater gone, took {self.visitor['taken']}")
            self.dashboard.log(f"Left with {self.visitor['taken']}")
        self.visitor = None
        self.watching = self.welcomed = False
        if self.reopen_at is None:
            self.plushie.eyes("CALM")

    def tick(self, now, hand_near, hand_seen, busy):
        """Every frame: the welcome, eyes on approach, the lid back open after a scare, the visitor leaving."""
        # Someone in view who hasn't reached the box yet: tell them the rules once.
        if hand_seen and not self.welcomed and self.visitor is None and self.reopen_at is None:
            self.welcomed = True
            self.last_hand = now
            self.roaster.say_welcome()
            self.dashboard.log("Welcomed someone")
        if hand_near:
            self.last_hand = now
            if not self.watching and self.reopen_at is None:
                self.watching = True
                self.plushie.eyes("SUSPICIOUS")  # I see you
        # The lid stays shut on them for the whole roast, and opens once it's been said.
        if self.reopen_at is not None and (self.roast_done.is_set() or now >= self.reopen_at):
            self.reopen_at = self.roast_done = None
            self.plushie.lid(closed=False)
            self.plushie.eyes("CALM")
        gone = now - self.last_hand > VISITOR_GONE_SECONDS
        if hand_seen and self.welcomed:
            self.last_hand = now  # still around (somewhere in view): don't welcome them again
        if gone and not busy and (self.visitor is not None or self.watching or self.welcomed):
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
        cv2.putText(frame, "Show the box's markers, or press Draw box on the dashboard", (10, 30),
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
    if DEMO_PIECES:
        print(f"DEMO_PIECES={DEMO_PIECES}: every grab counts as {DEMO_PIECES} pieces (not counted for real).")
    # API calls run on worker threads so the video never freezes while we wait.
    executor = ThreadPoolExecutor(max_workers=3)
    counts = []                # grabs being counted: (future, asked at)
    finding = None             # the box being looked for by box_finder: its future
    auto_box = counting_enabled  # keep finding the box by itself; a box drawn by hand turns it off, Find box on
    find_soon = auto_box       # look for the box as soon as the picture's up
    last_find = -REFIND_SECONDS
    black_since = None         # when the picture went black (the phone app in the background, or moved)
    last_count = "-" if counting_enabled else "counting off"

    tracker = BoxTracker()
    # A box found or drawn, used when there are no markers. Last time's is only used when it won't be looked for
    # (no OpenAI key): the camera has usually moved since.
    manual = None if auto_box else load_region()
    use_markers = True
    watcher = BowlWatcher()
    before = None              # the box with no hand near it: the "before" photo of the next grab
    clear = 0                  # frames in a row with no hand near the box
    grab = None                # the grab being finished: {"before", "since_exit", "hand": (sharpness, crop)}
    outline = None             # where the box is (the last frame's, for the dashboard's buttons)

    start = time.monotonic()
    last_ts = -1
    last_frame_time = start
    last_look = 0.0
    fps = 0.0
    frame_index = 0

    webbrowser.open(f"http://localhost:{DASHBOARD_PORT}")
    print("Running. The dashboard is the interface; Ctrl+C here stops it.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed, retrying...")
                time.sleep(0.2)
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
                    show.force_scare(frame, now, outline)
                elif action == "roast":  # roast whoever's in front of the camera, right now
                    roaster.roast(roaster.prepare(hide_box(frame, outline)), wait=12.0,
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
                elif action == "set_box":  # dragged over the camera on the dashboard
                    manual, use_markers, auto_box = save_region(tuple(value)), False, False
                    watcher, grab, before, clear = BowlWatcher(), None, None, 0
                    dashboard.log("Box drawn (Find box goes back to finding it by itself)")
                elif action == "find_box":  # ask OpenAI where the box is (box_finder.py), and keep doing so
                    find_soon = auto_box = counting_enabled
                elif action == "use_markers":  # back to the markers, looked for afresh
                    tracker.forget()
                    use_markers = True
                    watcher, grab, before, clear = BowlWatcher(), None, None, 0
                    dashboard.log("Looking for the box's markers")

            # Where the box is: its markers (looked for every few frames), else the hand-drawn box.
            if use_markers and frame_index % MARKER_EVERY == 1:
                tracker.update(frame)
            outline = tracker.outline() if use_markers else None
            source = f"markers {tracker.visible}/4" if outline is not None else "drawn"
            if outline is None:  # None at startup until the box is found: last time's box isn't trusted
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
                    show.reach(frame, outline)
                elif event == "LINGER":
                    print(f"[{now - start:7.1f}s] LINGER")
                    show.scare(now, f"hand in the box {LINGER_SECONDS:.0f}s")
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
                        if DEMO_PIECES:
                            last_count = f"{DEMO_PIECES} (demo)"
                            show.took(DEMO_PIECES, now)
                        elif counting_enabled:
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

            # Finding the box without markers, all the time: as soon as the picture's up, then every
            # REFIND_SECONDS, and at once after a black spell (the phone was probably moved). Only while nobody's
            # at the box (a hand hides it) and not during a catch (the lid's down over it).
            black = frame.mean() < BLACK_FRAME_MEAN
            if black:
                black_since = black_since or now
            else:
                if black_since is not None and now - black_since >= 2.0:
                    find_soon = auto_box
                black_since = None
            if auto_box and now - last_find >= REFIND_SECONDS:
                find_soon = True
            idle = not near and not watcher.inside and grab is None and show.reopen_at is None
            if (find_soon and finding is None and not black and frame_index > 15 and idle
                    and not (use_markers and tracker.outline() is not None)):
                find_soon, last_find = False, now
                finding = executor.submit(find_box, frame.copy())
            if finding is not None and finding.done():
                try:
                    found = finding.result()
                except Exception as e:
                    found = None
                    print(f"[{now - start:7.1f}s] couldn't find the box: {e}")
                finding = None
                current = tuple(manual.min(axis=0)) + tuple(manual.max(axis=0)) if manual is not None else None
                if found and (current is None or overlap(found, current) < BOX_MOVED_IOU):
                    if idle:  # moving it mid-grab would compare the wrong photos; the next look will do it
                        manual = save_region(found)
                        watcher, grab, before, clear = BowlWatcher(), None, None, 0
                        print(f"[{now - start:7.1f}s] box found at {tuple(round(v, 2) for v in found)}")
                        dashboard.log("Box found" if current is None else "Box moved: followed it")
                elif not found and manual is None:
                    # No box yet: nothing can set off a catch meanwhile, and it looks again soon.
                    last_find = now - REFIND_SECONDS + NO_BOX_RETRY_SECONDS
                    dashboard.log("Can't see the box yet: trying again (or Draw box)")

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

            show.tick(now, hand_near=near, hand_seen=bool(hands), busy=grab is not None or bool(counts))

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
                counting=counting_enabled, catching=show.reopen_at is not None, finding_box=finding is not None)
            draw_overlay(frame, outline, source, watcher, result, now, fps, status)
            dashboard.show(frame, now)
            if VIDEO_WINDOW:
                cv2.imshow("TreatOps", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        roaster.close()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
