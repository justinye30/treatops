"""
TreatOps' web page: the camera live, TOP BIG BACKS (the greediest trick-or-treaters, with their photo
and how much they took), a replay of every roast (the camera from just before the scare, with the roast's
voice), and buttons to drive the plushie.

hand_in_bowl.py starts it and feeds it; open http://localhost:8080 (DASHBOARD_PORT). It's only reachable
from this laptop unless DASHBOARD_HOST=0.0.0.0: it shows photos of trick-or-treaters, so open it up only on
a network you trust. Everything it keeps is in laptop/data/ (gitignored): delete it after the event.

Standard library only: an MJPEG stream for the camera, JSON for the rest, polled by web/index.html. The
page's push-to-talk sends WAV to POST /api/say, and the plushie plays it.
"""
import collections
import http.server
import json
import os
import queue
import shutil
import threading
import time
from pathlib import Path

import cv2

HERE = Path(__file__).parent
DATA = HERE / "data"
WEB = HERE / "web"
BOARD_PATH = DATA / "leaderboard.json"

PORT = int(os.environ.get("DASHBOARD_PORT") or 8080)
HOST = os.environ.get("DASHBOARD_HOST") or "127.0.0.1"
STREAM_FPS = 15
STREAM_WIDTH = 960
REPLAY_FPS = 10
REPLAY_WIDTH = 640
REPLAY_BEFORE = 2.0  # seconds of camera from before the scare
REPLAY_AFTER = 6.0   # ...and after
PHOTO_WIDTH = 480
EVENTS = 40
MAX_RECORDING_BYTES = 2_000_000  # about a minute of the mic at 16 kHz

# What the buttons can ask for; hand_in_bowl.py carries them out on its own thread.
ACTIONS = {"scare", "roast", "end_visitor", "lid", "eyes", "mute", "clear_board", "set_box", "find_box", "use_markers"}
MOODS = {"CALM", "SUSPICIOUS", "PLEASED", "ANGRY"}
TYPES = {".jpg": "image/jpeg", ".mp3": "audio/mpeg", ".json": "application/json", ".html": "text/html; charset=utf-8"}


def valid_box(value):
    """A box drawn on the page: [x1, y1, x2, y2] as 0-1 fractions of the camera picture, not a sliver."""
    if not isinstance(value, list) or len(value) != 4 or not all(isinstance(v, (int, float)) for v in value):
        return False
    x1, y1, x2, y2 = value
    return 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 and x2 - x1 >= 0.03 and y2 - y1 >= 0.03


def _jpeg(frame, width, quality):
    if frame is None or frame.size == 0:
        return None
    if frame.shape[1] > width:
        frame = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return data.tobytes() if ok else None


class Dashboard:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition()
        self._stream_jpeg = None
        self._last_stream = 0.0
        self._viewers = 0
        self._last_sample = 0.0
        self._ring = collections.deque(maxlen=int(REPLAY_BEFORE * REPLAY_FPS))  # the last few seconds, for replays
        self._recording = None  # {"id", "frames", "until", "started"}
        self._events = collections.deque(maxlen=EVENTS)
        self._controls = queue.Queue()
        self.status = {}
        DATA.mkdir(exist_ok=True)
        self.board = json.loads(BOARD_PATH.read_text()) if BOARD_PATH.exists() else {}
        self._serve()
        print(f"Dashboard: http://localhost:{PORT}" + (f" (and http://<this laptop>:{PORT})" if HOST != "127.0.0.1" else ""))

    # ---------- Fed by the main loop ----------

    def feed(self, frame, now):
        """Every camera frame, before anything's drawn on it: kept for replays."""
        if now - self._last_sample < 1.0 / REPLAY_FPS:
            return
        self._last_sample = now
        jpeg = _jpeg(frame, REPLAY_WIDTH, 80)
        if jpeg is None:
            return
        self._ring.append(jpeg)
        if self._recording:
            self._recording["frames"].append(jpeg)
            if now >= self._recording["until"]:
                recording, self._recording = self._recording, None
                threading.Thread(target=self._save_replay, args=(recording,), daemon=True).start()

    def show(self, frame, now):
        """Every frame with its overlay drawn: the live stream, while someone's watching."""
        if not self._viewers or now - self._last_stream < 1.0 / STREAM_FPS:
            return
        self._last_stream = now
        jpeg = _jpeg(frame, STREAM_WIDTH, 70)
        with self._frame_ready:
            self._stream_jpeg = jpeg
            self._frame_ready.notify_all()

    def set_status(self, **status):
        self.status = status

    def log(self, text):
        self._events.appendleft({"at": time.time(), "text": text})

    def controls(self):
        """The buttons pressed since last time: (action, value) pairs."""
        pressed = []
        while not self._controls.empty():
            pressed.append(self._controls.get_nowait())
        return pressed

    def visitor_started(self, visitor_id, frame):
        """A new trick-or-treater; `frame` is the photo of their costume (None if there isn't one)."""
        DATA.joinpath("photos").mkdir(exist_ok=True)
        photo = _jpeg(frame, PHOTO_WIDTH, 85) if frame is not None else None
        if photo:
            (DATA / "photos" / f"{visitor_id}.jpg").write_bytes(photo)
        with self._lock:
            self.board[visitor_id] = {"id": visitor_id, "at": time.time(), "taken": 0, "caught": False,
                                      "photo": bool(photo), "roast": None, "replay": None}
            self._save_board()

    def visitor_took(self, visitor_id, taken):
        with self._lock:
            if visitor_id in self.board:
                self.board[visitor_id]["taken"] = taken
                self._save_board()

    def caught(self, visitor_id, now):
        """The scare: start recording the replay, from the seconds just before it."""
        with self._lock:
            if visitor_id in self.board:
                self.board[visitor_id]["caught"] = True
                self._save_board()
        if self._recording is None:
            started = now - len(self._ring) / REPLAY_FPS
            self._recording = {"id": visitor_id, "frames": list(self._ring), "until": now + REPLAY_AFTER,
                               "started": started, "audio_at": None}

    def roast_spoken(self, visitor_id, text, mp3):
        """The roast as it's about to be said: its words for the leaderboard, its voice for the replay."""
        now = time.monotonic()
        folder = DATA / "replays" / visitor_id
        folder.mkdir(parents=True, exist_ok=True)
        if mp3:
            (folder / "roast.mp3").write_bytes(mp3)
        recording = self._recording
        if recording and recording["id"] == visitor_id:
            recording["audio_at"] = now - recording["started"]  # where the voice starts in the replay
        with self._lock:
            entry = self.board.get(visitor_id)
            if entry:
                entry["roast"] = text
                if entry["replay"] is not None and mp3:  # the replay was saved first
                    entry["replay"]["audio"] = True
                self._save_board()

    def clear(self):
        with self._lock:
            self.board = {}
            self._save_board()
        for folder in ("photos", "replays", "grabs"):  # grabs: photos an earlier version kept
            shutil.rmtree(DATA / folder, ignore_errors=True)
        self.log("Leaderboard cleared")

    # ---------- Keeping it ----------

    def _save_board(self):
        """Called with the lock held. Written aside and moved in, so a crash never leaves half a file."""
        partial = BOARD_PATH.with_suffix(".part")
        partial.write_text(json.dumps(self.board, indent=1))
        os.replace(partial, BOARD_PATH)

    def _save_replay(self, recording):
        folder = DATA / "replays" / recording["id"]
        folder.mkdir(parents=True, exist_ok=True)
        for i, jpeg in enumerate(recording["frames"]):
            (folder / f"{i:04d}.jpg").write_bytes(jpeg)
        with self._lock:
            entry = self.board.get(recording["id"])
            if entry:
                entry["replay"] = {"frames": len(recording["frames"]), "fps": REPLAY_FPS,
                                   "audio": (folder / "roast.mp3").exists(),
                                   "audio_at": round(recording["audio_at"] or REPLAY_BEFORE, 2)}
                self._save_board()
        print(f"  replay saved ({len(recording['frames'])} frames)")

    # ---------- The web side ----------

    def state(self):
        with self._lock:
            entries = list(self.board.values())
        # Big backs only: two pieces or more (one piece is a good choice, not a big back).
        leaders = sorted((e for e in entries if e["taken"] >= 2), key=lambda e: (-e["taken"], -e["at"]))
        replays = sorted((e for e in entries if e["replay"]), key=lambda e: -e["at"])
        return {
            "status": self.status,
            "candies": sum(e["taken"] for e in entries),
            "visitors": len(entries),
            "board": leaders[:20],
            "replays": replays[:24],
            "events": list(self._events),
        }

    def _serve(self):
        dashboard = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    self._file(WEB / "index.html")
                elif path == "/stream.mjpg":
                    self._stream()
                elif path == "/api/state":
                    self._json(dashboard.state())
                elif path.startswith("/data/"):
                    target = (DATA / path[len("/data/"):]).resolve()
                    if DATA.resolve() in target.parents and target.is_file():
                        self._file(target)
                    else:
                        self.send_error(404)
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path == "/api/say":
                    self._say()
                    return
                if self.path != "/api/control":
                    self.send_error(404)
                    return
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                except ValueError:
                    self.send_error(400)
                    return
                action, value = body.get("action"), body.get("value")
                if (action not in ACTIONS or (action == "eyes" and value not in MOODS)
                        or (action == "set_box" and not valid_box(value))):
                    self.send_error(400)
                    return
                dashboard._controls.put((action, value))
                self._json({"ok": True})

            def _say(self):
                """A recording from the dashboard's mic (16-bit WAV), for the plushie to play."""
                length = int(self.headers.get("Content-Length") or 0)
                if not 44 < length <= MAX_RECORDING_BYTES:
                    self.send_error(413 if length > MAX_RECORDING_BYTES else 400)
                    return
                wav = self.rfile.read(length)
                if wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
                    self.send_error(415)
                    return
                dashboard._controls.put(("say", wav))
                self._json({"ok": True})

            def _json(self, data):
                payload = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _file(self, path):
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", TYPES.get(path.suffix, "application/octet-stream"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with dashboard._lock:
                    dashboard._viewers += 1
                last = None
                try:
                    while True:
                        with dashboard._frame_ready:
                            dashboard._frame_ready.wait(timeout=1.0)
                            jpeg = dashboard._stream_jpeg
                        if jpeg is None or jpeg is last:
                            continue
                        last = jpeg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # the page was closed
                finally:
                    with dashboard._lock:
                        dashboard._viewers -= 1

            def log_message(self, *args):
                pass  # quiet

        server = http.server.ThreadingHTTPServer((HOST, PORT), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
