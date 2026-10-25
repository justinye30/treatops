"""
Find the candy box in a camera frame by asking OpenAI where it is: no markers, nothing to draw.

hand_in_bowl.py runs it at startup, whenever the camera comes back after going black (the phone was probably
moved), and when the dashboard's "Find box" is pressed. It takes a few seconds and the fit is loose, so the
dashboard's "Draw box" is there to tidy it up. Markers on the box's corners, when seen, still win.

Needs OPENAI_API_KEY in laptop/.env. To try it on a photo:  python box_finder.py photo.jpg
"""
import base64
import json
import re
import sys
from pathlib import Path

import cv2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).with_name(".env"))

MODEL = "gpt-6-luna"
TIMEOUT_SECONDS = 20.0
MAX_SIDE = 1024
MIN_SIDE = 0.05  # a box smaller than this share of the picture is a wrong answer, not a box

INSTRUCTIONS = (
    "Find the open candy box (the container holding the Halloween candy) in this camera image. Reply with ONLY "
    'JSON: {"found": true, "x1": ..., "y1": ..., "x2": ..., "y2": ...} giving the bounding box of the box\'s '
    "opening, where the candy is, as fractions of the image width and height (0 to 1, x1 < x2, y1 < y2), or "
    '{"found": false} if there is no candy box. Ignore any rectangles, stamps or text drawn over the image.'
)

_client = None


def find_box(frame):
    """Where the candy box is in this BGR frame: (x1, y1, x2, y2) as 0-1 fractions, or None. Raises on API errors."""
    global _client
    if _client is None:
        _client = OpenAI(timeout=TIMEOUT_SECONDS, max_retries=0)
    scale = MAX_SIDE / max(frame.shape[:2])
    if scale < 1:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Could not encode image as JPEG")
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")
    response = _client.responses.create(
        model=MODEL,
        reasoning={"effort": "low"},
        max_output_tokens=300,
        instructions=INSTRUCTIONS,
        input=[{"role": "user", "content": [{"type": "input_image", "image_url": data_url}]}],
    )
    match = re.search(r"\{.*\}", response.output_text, re.S)
    if not match:
        return None
    answer = json.loads(match.group(0))
    if not answer.get("found"):
        return None
    x1, y1, x2, y2 = (min(1.0, max(0.0, float(answer[k]))) for k in ("x1", "y1", "x2", "y2"))
    if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
        return None
    return x1, y1, x2, y2


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python box_finder.py photo.jpg")
    image = cv2.imread(sys.argv[1])
    if image is None:
        raise SystemExit(f"Could not read {sys.argv[1]}")
    print(find_box(image) or "No candy box found")
