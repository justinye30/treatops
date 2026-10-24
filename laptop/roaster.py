"""
Heckle a trick-or-treater about their costume, out loud through the plushie.

prepare(frame) starts writing the roast (OpenAI) and turning it into speech (ElevenLabs) in the background
the moment someone reaches into the box, so it's ready by the time we know they took too much: the two calls
take a few seconds together. say_warning() and the fallbacks speak canned lines, synthesised once and cached
in voice_cache/. Everything is spoken through plushie.say() (see plushie.py).

Needs OPENAI_API_KEY and ELEVENLABS_API_KEY in laptop/.env. Without ElevenLabs the lines are printed instead
of spoken; without OpenAI every roast is a canned one.

To try it on a photo:  python roaster.py photo.jpg   (plays on this laptop)
"""
import base64
import hashlib
import json
import os
import queue
import random
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

MODEL = "gpt-6-luna"
ROAST_TIMEOUT_SECONDS = 8.0
VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID") or "JBFqnCBsd6RMkjVDRZzb"  # any ElevenLabs voice works
VOICE_MODEL = "eleven_flash_v2_5"  # their fastest
VOICE_FORMAT = "mp3_22050_32"      # small MP3s the ESP32 streams and decodes easily
CACHE = HERE / "voice_cache"

ROAST_INSTRUCTIONS = (
    "You're a plushie heckling trick-or-treaters from next to a Halloween candy box. This one just grabbed "
    "way too much candy. Heckle them about their costume in the photo, like the loudest kid in the back of "
    "the class. Write ONE line of at most 15 words that sounds shouted, not written. Use Gen Alpha brainrot "
    "slang the way kids actually talk (nah, bro, not you, the audacity, caught in 4K, fanum tax, aura, NPC, "
    "it's giving, ain't no way), one or two bits, not a pile. Name the costume and dunk on it: cheap, "
    "dollar store, Temu, low effort, falling apart, last-minute, wrong size. Tie it to grabbing extra "
    "candy. No poetry, rhymes, metaphors, puns, emojis, hashtags or quote marks. Keep it PG for kids: no "
    "swearing, and nothing about their body, face, skin, hair, weight, age, gender, race or religion; only "
    "the costume and the greed. If you can't make out a costume, roast the greed.\n"
    "Examples of the tone:\n"
    "Not you in that dollar store mask trying to grab extra candy.\n"
    "Bro's Spider-Man suit is straight off Temu and he still took three. Nah.\n"
    "Caught in 4K, plastic witch hat, triple candy. Negative aura."
)

# Two pieces: a warning shot.
WARNINGS = [
    "Nah bro, that's two. I'm watching you.",
    "Two? Easy there, fanum tax collector.",
    "Bro thinks I didn't see that. Put it back.",
]

# Three or more when the roast isn't ready or the API is down.
FALLBACK_ROASTS = [
    "Not you trying to fanum tax my whole box.",
    "Caught in 4K taking three. Negative aura, bro.",
    "Ain't no way you just did that in that costume.",
    "Bro's costume is from Temu and so are his manners.",
    "The audacity. Put it back, NPC.",
    "Nah, that's not trick-or-treat, that's a robbery.",
    "Low effort costume, high effort greed. Crazy.",
    "Bro really said one for me and three for me.",
    "It's giving last-minute costume and first-place greed.",
    "Somebody call your mom, bro's stealing candy.",
]

_openai = None


def _client():
    global _openai
    if _openai is None:
        _openai = OpenAI(timeout=ROAST_TIMEOUT_SECONDS, max_retries=0)
    return _openai


def write_roast(frame):
    """One heckle about the costume in this BGR photo."""
    scale = 768 / max(frame.shape[:2])
    if scale < 1:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Could not encode image as JPEG")
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")
    response = _client().responses.create(
        model=MODEL,
        reasoning={"effort": "none"},  # it's one line; thinking only adds seconds
        max_output_tokens=80,
        instructions=ROAST_INSTRUCTIONS,
        input=[{"role": "user", "content": [{"type": "input_image", "image_url": data_url}]}],
    )
    text = response.output_text.strip().strip('"')
    # Never say a refusal or a line cut off mid-word out loud: the caller falls back to a canned roast.
    if not text or len(text.split()) > 25 or text.lower().startswith(("i can't", "i cannot", "sorry", "i'm sorry")):
        raise ValueError(f"unusable roast: {text!r}")
    return text


def synthesize(text):
    """The line spoken by ElevenLabs, as MP3 bytes. Raises if the call fails."""
    request = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}?output_format={VOICE_FORMAT}",
        data=json.dumps({"text": text, "model_id": VOICE_MODEL}).encode(),
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read()


class Roaster:
    def __init__(self, plushie):
        self.plushie = plushie
        self.can_roast = bool(os.environ.get("OPENAI_API_KEY"))
        self.can_speak = bool(os.environ.get("ELEVENLABS_API_KEY"))
        if not self.can_speak:
            print("ELEVENLABS_API_KEY not set (see laptop/.env): lines are printed, not spoken.")
        self._work = ThreadPoolExecutor(max_workers=2)
        self._lines = queue.Queue()  # (text, mp3 or None), spoken one at a time
        threading.Thread(target=self._speaker, daemon=True).start()
        if self.can_speak:
            # Synthesise the canned lines now, so a fallback never waits on the network. One job per line,
            # so quitting can cancel the ones not started.
            for line in WARNINGS + FALLBACK_ROASTS:
                self._work.submit(self._canned, line)

    def close(self):
        self._work.shutdown(wait=False, cancel_futures=True)

    def prepare(self, frame):
        """Starts the roast for the costume in this frame. Returns a future of (text, mp3 or None)."""
        if not self.can_roast:
            return None
        frame = frame.copy()

        def make():
            text = write_roast(frame)
            print(f"  roast ready: {text}")
            return text, (synthesize(text) if self.can_speak else None)

        return self._work.submit(make)

    def roast(self, prepared, wait):
        """Speaks the prepared roast, waiting up to `wait` seconds for it; a canned one if it isn't ready."""
        def run():
            line = None
            if prepared is not None:
                try:
                    line = prepared.result(timeout=wait)
                except Exception as error:  # still being written, or the API failed
                    print(f"  roast not ready ({error.__class__.__name__}), using a canned one")
            self._lines.put(line or self._canned(random.choice(FALLBACK_ROASTS)))
        threading.Thread(target=run, daemon=True).start()

    def say_warning(self):
        # Off the caller's thread: an uncached line means a call to ElevenLabs, and that mustn't stall the video.
        threading.Thread(target=lambda: self._lines.put(self._canned(random.choice(WARNINGS))), daemon=True).start()

    def _canned(self, text):
        """A canned line with its audio from voice_cache/, synthesised the first time it's needed."""
        if not self.can_speak:
            return text, None
        key = hashlib.sha1(f"{VOICE_ID}|{VOICE_MODEL}|{VOICE_FORMAT}|{text}".encode()).hexdigest()[:16]
        path = CACHE / f"{key}.mp3"
        if not path.exists():
            try:
                CACHE.mkdir(exist_ok=True)
                # Written aside, then moved in: another thread reading it never sees half a file.
                partial = path.with_suffix(f".{threading.get_ident()}.part")
                partial.write_bytes(synthesize(text))
                os.replace(partial, path)
            except Exception as error:
                print(f"  couldn't synthesise a canned line: {error}")
                return text, None
        return text, path.read_bytes()

    def _speaker(self):
        while True:
            text, mp3 = self._lines.get()
            print(f"  PLUSHIE SAYS: {text}")
            if mp3:
                try:
                    self.plushie.say(mp3)
                except Exception as error:
                    print(f"  couldn't play the line: {error}")


if __name__ == "__main__":
    from plushie import play_locally

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python roaster.py photo.jpg")
    image = cv2.imread(sys.argv[1])
    if image is None:
        raise SystemExit(f"Could not read {sys.argv[1]}")
    line = write_roast(image)
    print(line)
    if os.environ.get("ELEVENLABS_API_KEY"):
        play_locally(synthesize(line))
