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
import time
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
VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID") or "8ZS1KKN5BCOVHpGCtbLx"  # "Gorak - Powerful, Deep", an orc
VOICE_SPEED = float(os.environ.get("ELEVENLABS_SPEED") or 1.2)  # 0.7-1.2; 1.2 is their fastest talking
VOICE_MODEL = "eleven_flash_v2_5"  # their fastest
VOICE_FORMAT = "mp3_22050_32"      # small MP3s the ESP32 streams and decodes easily
CACHE = HERE / "voice_cache"

ROAST_INSTRUCTIONS = (
    "You're a grumpy plushie guarding a Halloween candy box, like an unimpressed security guard who takes "
    "the job far too seriously. This trick-or-treater just grabbed way too much candy. Tease them about "
    "their costume in the photo: dry, deadpan, a little dramatic, funny rather than mean. Write ONE line of "
    "at most 15 words, to be said out loud: a quick setup and a punchline, not a description. Never "
    "narrate what you see (no 'I see'). Name the costume and poke fun at it (homemade, cheap, falling "
    "apart, last-minute, wrong size), and tie it to taking too much candy, without saying how many pieces. Plain everyday English: no "
    "internet or Gen Alpha slang (no bro, nah, NPC, aura, rizz, caught in 4K, it's giving, no cap, fanum "
    "tax, the audacity). No rhymes, emojis, hashtags or quote marks. Keep it PG for kids: no swearing, and "
    "nothing about their body, face, skin, hair, weight, age, gender, race or religion; only the costume "
    "and the greed. The camera looks down at the cardboard box the candy sits in: that box, the candy and "
    "any gray patch painted over them are never the costume, so never mention a box. Look only at the "
    "person: their sleeves, hands, gloves, clothes, mask or props. No costume at all (everyday clothes) is a joke in itself: dressed as a regular person, "
    "the costume is commitment issues, and so on.\n"
    "Examples of the tone:\n"
    "A cardboard robot with a candy problem. Even your costume looks disappointed.\n"
    "That witch hat is held together with tape and hope. Put two back.\n"
    "Spider-Man takes one piece. You are not Spider-Man."
)

# Someone's coming, before they've reached the box.
WELCOMES = [
    "Take some candy, but not too much.",
    "One piece is fine. I'm watching you.",
    "Go ahead, grab one. Just one.",
]

# Two pieces: a warning shot.
WARNINGS = [
    "That's two. I'm counting, you know.",
    "Two pieces? Bold. Very bold.",
    "You think I didn't see that? Put it back.",
]

# Three or more when the roast isn't ready or the API is down.
FALLBACK_ROASTS = [
    "That's not trick-or-treating. That's a heist.",
    "I've seen raccoons with more self-control.",
    "Put it back. Slowly. Nobody has to get hurt.",
    "Leave some for the rest of the neighborhood, please.",
    "Your costume says hero. Your hands say candy thief.",
    "Right in front of me? Impressive, honestly.",
    "I'm a plushie, not a pushover. Put some back.",
    "Hold it right there, candy bandit.",
    "The sign said one. I watched you read the sign.",
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
        data=json.dumps({"text": text, "model_id": VOICE_MODEL, "voice_settings": {"speed": VOICE_SPEED}}).encode(),
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read()


class Roaster:
    def __init__(self, plushie):
        self.plushie = plushie
        self.can_roast = bool(os.environ.get("OPENAI_API_KEY"))
        self.can_speak = bool(os.environ.get("ELEVENLABS_API_KEY"))
        self.muted = False  # the dashboard's mute button: lines are still written and kept, just not played
        self.speaking = None  # the line being said right now (the dashboard subtitles it)
        if not self.can_speak:
            print("ELEVENLABS_API_KEY not set (see laptop/.env): lines are printed, not spoken.")
        self._work = ThreadPoolExecutor(max_workers=2)
        self._lines = queue.Queue()  # (text, mp3 or None), spoken one at a time
        threading.Thread(target=self._speaker, daemon=True).start()
        if self.can_speak:
            # Synthesise the canned lines now, so a fallback never waits on the network. One job per line,
            # so quitting can cancel the ones not started.
            for line in WELCOMES + WARNINGS + FALLBACK_ROASTS:
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

    def roast(self, prepared, wait, on_spoken=None):
        """
        Speaks the prepared roast, waiting up to `wait` seconds for it; a canned one if it isn't ready.
        on_spoken(text, mp3) hears what's said just before it's said (the dashboard keeps it).
        Returns an Event that's set once the roast has finished playing (the lid opens then).
        """
        done = threading.Event()

        def run():
            line = None
            try:
                if prepared is not None:
                    try:
                        line = prepared.result(timeout=wait)
                    except Exception as error:  # still being written, or the API failed
                        print(f"  roast not ready ({error.__class__.__name__}), using a canned one")
                line = line or self._canned(random.choice(FALLBACK_ROASTS))
                if on_spoken:
                    try:
                        on_spoken(*line)
                    except Exception as error:
                        print(f"  couldn't keep the roast: {error}")
                self._lines.put((*line, "mp3", done))
            except Exception as error:
                print(f"  couldn't roast: {error}")
                done.set()
        threading.Thread(target=run, daemon=True).start()
        return done

    def say_welcome(self):
        threading.Thread(target=lambda: self._lines.put(self._canned(random.choice(WELCOMES))), daemon=True).start()

    def say_warning(self):
        # Off the caller's thread: an uncached line means a call to ElevenLabs, and that mustn't stall the video.
        threading.Thread(target=lambda: self._lines.put(self._canned(random.choice(WARNINGS))), daemon=True).start()

    def say_recording(self, wav):
        """The operator talking through the dashboard's mic: played in turn with the lines, even when muted."""
        self._lines.put(("Operator on the speaker", wav, "wav"))

    def _canned(self, text):
        """A canned line with its audio from voice_cache/, synthesised the first time it's needed."""
        if not self.can_speak:
            return text, None
        key = hashlib.sha1(f"{VOICE_ID}|{VOICE_SPEED}|{VOICE_MODEL}|{VOICE_FORMAT}|{text}".encode()).hexdigest()[:16]
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
            # (text, mp3) for the lines; (text, wav, "wav") for the operator's mic, which plays even when muted;
            # a roast also carries an Event, set once it's been said.
            text, audio, *rest = self._lines.get()
            kind = rest[0] if rest else "mp3"
            done = rest[1] if len(rest) > 1 else None
            muted = self.muted and kind == "mp3"
            print(f"  PLUSHIE SAYS: {text}" + ("  (muted)" if muted else ""))
            self.speaking = text
            try:
                if audio and not muted:
                    self.plushie.say(audio, kind)
                else:
                    time.sleep(3)  # long enough to read the subtitle
            except Exception as error:
                print(f"  couldn't play the line: {error}")
            finally:
                self.speaking = None
                if done:
                    done.set()


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
