"""
The plushie: an ESP32 with a speaker, on the same Wi-Fi as this laptop.

The laptop does the thinking (OpenAI, ElevenLabs) and the plushie does the talking. say(mp3) keeps the clip
in a small web server here and asks the ESP32 to stream it and play it (esp32/candy_cop). Commands such as
LID CLOSE go to the ESP32 the same way:

  LID OPEN | LID CLOSE                   the cardboard lid
  EYES CALM|SUSPICIOUS|PLEASED|ANGRY     the eyes' mood
  LOOK x y                               look that way (-1..1 each, as the plushie's face is seen)
  COUNT n                                flash the victims counter

Set PLUSHIE_HOST in laptop/.env to the ESP32's IP (it prints it on boot) or candycop.local. Without it the
voice plays on this laptop's speakers and commands are only printed, so everything can be tested on a desk.
If the venue's Wi-Fi keeps devices apart, put the laptop and the ESP32 on a phone hotspot.
"""
import ctypes
import http.server
import itertools
import os
import platform
import queue
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from collections import OrderedDict

VOICE_PORT = int(os.environ.get("VOICE_PORT") or 8765)  # where the ESP32 fetches clips from this laptop
MP3_BYTES_PER_SECOND = 32000 / 8                        # ElevenLabs' mp3_22050_32: 32 kbit/s
KEEP_CLIPS = 8


class Plushie:
    def __init__(self, host=None):
        self.host = host or None
        self._clips = OrderedDict()  # name -> MP3 bytes, served to the ESP32
        self._names = itertools.count()
        self._commands = queue.Queue()
        # Commands go out on their own thread: a slow or missing ESP32 must never stall the video.
        threading.Thread(target=self._send_commands, daemon=True).start()
        if self.host:
            self._serve()
            print(f"Plushie at {self.host}; serving its voice on port {VOICE_PORT}.")
        else:
            print("PLUSHIE_HOST not set (see laptop/.env): voice on this laptop, commands printed only.")

    def say(self, mp3):
        """Plays an MP3 clip through the plushie (or this laptop) and returns about when it's done."""
        seconds = len(mp3) / MP3_BYTES_PER_SECOND
        if self.host:
            name = f"line-{next(self._names)}.mp3"
            self._clips[name] = mp3
            while len(self._clips) > KEEP_CLIPS:
                self._clips.popitem(last=False)
            try:
                # It answers once it has the stream going (or fails to: then it plays here instead).
                self._get("/play", timeout=4, file=name, port=VOICE_PORT)
                time.sleep(seconds + 0.8)  # it plays as it streams; wait it out so lines don't cut each other off
                return
            except Exception as error:
                print(f"  plushie couldn't play it ({error}); playing on the laptop instead")
        play_locally(mp3)

    def command(self, text):
        print(f"  -> plushie: {text}")
        self._commands.put(text)

    def lid(self, closed):
        self.command("LID CLOSE" if closed else "LID OPEN")

    def eyes(self, mood):
        self.command(f"EYES {mood}")

    def look(self, x, y):
        """Where to look, -1..1 each (right and down positive, as the face is seen). Not printed: it's often."""
        self._commands.put(f"LOOK {x:.2f} {y:.2f}")

    def count(self, victims):
        self.command(f"COUNT {victims}")

    def _send_commands(self):
        while True:
            text = self._commands.get()
            if not self.host:
                continue
            for attempt in range(3):  # a Wi-Fi hiccup mustn't leave the lid shut
                try:
                    self._get("/cmd", c=text)
                    break
                except Exception as error:
                    if attempt == 2:
                        print(f"  plushie didn't take '{text}': {error}")

    def _get(self, path, timeout=2, **params):
        url = f"http://{self.host}{path}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=timeout) as response:
            response.read()

    def _serve(self):
        clips = self._clips

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                mp3 = clips.get(self.path.lstrip("/"))
                if mp3 is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self.send_header("Content-Length", str(len(mp3)))
                self.end_headers()
                self.wfile.write(mp3)

            def log_message(self, *args):
                pass  # quiet

        server = http.server.ThreadingHTTPServer(("0.0.0.0", VOICE_PORT), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()


def play_locally(mp3):
    """Plays MP3 bytes on this computer's speakers and returns when it's done."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as file:
        file.write(mp3)
    try:
        system = platform.system()
        if system == "Windows":
            # Windows' built-in media player API: plays MP3 without installing anything.
            mci = ctypes.windll.winmm.mciSendStringW
            alias = f"clip{threading.get_ident()}"
            mci(f'open "{file.name}" type mpegvideo alias {alias}', None, 0, None)
            mci(f"play {alias} wait", None, 0, None)
            mci(f"close {alias}", None, 0, None)
        elif system == "Darwin":
            subprocess.run(["afplay", file.name], check=False)
        else:
            subprocess.run(["mpg123", "-q", file.name], check=False)
    finally:
        os.unlink(file.name)
