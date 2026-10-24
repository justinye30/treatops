"""
Count how many pieces of candy a hand is holding, using OpenAI's Decisions API.

Needs OPENAI_API_KEY, either in laptop/.env (copy .env.example) or in your environment.

hand_in_bowl.py calls count_candy() when a hand leaves the bowl. To try it on
a photo by itself:

  python candy_counter.py photo.jpg
"""
import base64
import sys
from pathlib import Path

import cv2
from dotenv import load_dotenv
from openai import OpenAI

# Pull OPENAI_API_KEY from laptop/.env. A key already set in the environment wins.
load_dotenv(Path(__file__).with_name(".env"))

MODEL = "gpt-6-luna"
TIMEOUT_SECONDS = 5.0  # give up rather than stall the scare
CROP_PADDING = 0.6     # grow the hand's box by this fraction per side so candy in the fist is kept
JPEG_QUALITY = 85

COUNT_QUESTION = {
    "type": "choice",
    "name": "candy_count",
    "instructions": (
        "This photo shows a trick-or-treater's hand right after it left a Halloween "
        "candy bowl. How many pieces of candy is the hand holding? Count each wrapped "
        "candy or candy bar as one piece. If fingers hide part of the candy, estimate "
        "from the wrappers you can see. Ignore candy still in the bowl."
    ),
    "choices": [
        {"value": "0", "description": "The hand is empty or holding something that isn't candy."},
        {"value": "1", "description": "Exactly one piece of candy."},
        {"value": "2", "description": "Exactly two pieces of candy."},
        {"value": "3+", "description": "Three or more pieces, or a whole handful."},
        {"value": "unclear", "description": "No hand is visible, or the image is too blurry or dark to tell."},
    ],
}

_client = None


def _get_client():
    global _client
    if _client is None:
        # No retries: a late answer is useless once the kid has walked away.
        _client = OpenAI(timeout=TIMEOUT_SECONDS, max_retries=0)
    return _client


def crop_to_hands(frame, hands, padding=CROP_PADDING):
    """Crop the frame to a padded box around every detected hand (MediaPipe landmark lists)."""
    xs = [lm.x for landmarks in hands for lm in landmarks]
    ys = [lm.y for landmarks in hands for lm in landmarks]
    pad_x = (max(xs) - min(xs)) * padding
    pad_y = (max(ys) - min(ys)) * padding
    fh, fw = frame.shape[:2]
    x1 = max(int((min(xs) - pad_x) * fw), 0)
    x2 = min(int((max(xs) + pad_x) * fw), fw)
    y1 = max(int((min(ys) - pad_y) * fh), 0)
    y2 = min(int((max(ys) + pad_y) * fh), fh)
    return frame[y1:y2, x1:x2]


def count_candy(image):
    """
    Ask the Decisions API how much candy is in the hand in this BGR image.
    Returns (choice, confidence), where choice is one of the COUNT_QUESTION values,
    or "refused". Network and API errors are raised to the caller.
    """
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("Could not encode image as JPEG")
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")

    decision = _get_client().decisions.create(
        model=MODEL,
        input=[{"role": "user", "content": [{"type": "input_image", "image_url": data_url}]}],
        questions=[COUNT_QUESTION],
    )

    answer = decision.answers[0]
    if answer.type == "refusal":
        return "refused", 0.0
    return answer.choice, answer.confidence


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python candy_counter.py photo.jpg")
    img = cv2.imread(sys.argv[1])
    if img is None:
        raise SystemExit(f"Could not read {sys.argv[1]}")
    choice, confidence = count_candy(img)
    print(f"Candy in hand: {choice} (confidence {confidence:.2f})")
