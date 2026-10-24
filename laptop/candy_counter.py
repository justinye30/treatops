"""
Count how much candy a trick-or-treater took, using OpenAI's Decisions API.

Needs OPENAI_API_KEY, either in laptop/.env (copy .env.example) or in your environment.

hand_in_bowl.py calls count_taken() after each grab with three photos: the box just before the hand went
in, the box once it's out again, and the sharpest look at the hand on its way out. Comparing the box is the
main evidence (a hand closes round what it holds); the hand backs it up. To try it on photos by themselves:

  python candy_counter.py before.jpg after.jpg [hand.jpg]
  python candy_counter.py hand.jpg              (just the hand, as before)
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
TIMEOUT_SECONDS = 8.0  # give up rather than stall the scare (three photos usually take 1-2 s)
CROP_PADDING = 0.6     # grow the hand's box by this fraction per side so candy in the fist is kept
JPEG_QUALITY = 90
MAX_SIDE = 1024        # larger photos are scaled down before sending

CHOICES = [
    {"value": "0", "description": "No candy was taken."},
    {"value": "1", "description": "Exactly one piece of candy."},
    {"value": "2", "description": "Exactly two pieces of candy."},
    {"value": "3+", "description": "Three or more pieces, or a whole handful."},
    {"value": "unclear", "description": "The photos are too blurry, dark or blocked to tell."},
]
PIECES = {"0": 0, "1": 1, "2": 2, "3+": 3}  # what each answer adds to a visitor's total

TAKEN_QUESTION = {
    "type": "choice",
    "name": "candy_taken",
    "instructions": (
        "Image 1 shows a Halloween candy box just before a trick-or-treater reached in, and image 2 shows "
        "the same box from the same camera right after their hand left. If there is an image 3, it is their "
        "hand on its way out of the box. How many pieces of candy did they take? Compare the two box photos: "
        "count each wrapped candy or candy bar that is in image 1 but gone from image 2. Candy that only "
        "moved around inside the box was not taken. Use the hand, if shown, to confirm: count the candy it "
        "holds, estimating from the wrappers you can see when fingers hide some."
    ),
    "choices": CHOICES,
}

COUNT_QUESTION = {
    "type": "choice",
    "name": "candy_count",
    "instructions": (
        "This photo shows a trick-or-treater's hand right after it left a Halloween candy box. How many "
        "pieces of candy is the hand holding? Count each wrapped candy or candy bar as one piece. If fingers "
        "hide part of the candy, estimate from the wrappers you can see. Ignore candy still in the box."
    ),
    "choices": CHOICES,
}

_client = None


def _get_client():
    global _client
    if _client is None:
        # No retries: a late answer is useless once the kid has walked away.
        _client = OpenAI(timeout=TIMEOUT_SECONDS, max_retries=0)
    return _client


def crop_to_hands(frame, hands, padding=CROP_PADDING):
    """Crop the frame to a padded box around every given hand (MediaPipe landmark lists)."""
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


def sharpness(image):
    """How sharp a photo is (variance of the Laplacian): a hand moving fast blurs and scores low."""
    if image is None or image.size == 0:
        return 0.0
    return float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def changed_fraction(before, after):
    """How much of the box looks different afterwards (0-1). A rough check that runs locally."""
    if before is None or after is None or before.size == 0 or after.size == 0:
        return 0.0
    after = cv2.resize(after, (before.shape[1], before.shape[0]))
    a = cv2.GaussianBlur(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    return float((cv2.absdiff(a, b) > 35).mean())


def _image_part(image):
    """A BGR image as an input_image for the API (a JPEG data URL), scaled down if it's large."""
    if image is None or image.size == 0:
        raise ValueError("Empty image")
    scale = MAX_SIDE / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("Could not encode image as JPEG")
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")
    return {"type": "input_image", "image_url": data_url, "detail": "high"}


def _ask(content, question):
    decision = _get_client().decisions.create(
        model=MODEL,
        input=[{"role": "user", "content": content}],
        questions=[question],
    )
    answer = decision.answers[0]
    if answer.type == "refusal":
        return "refused", 0.0
    return answer.choice, answer.confidence


def count_taken(before, after, hand=None):
    """
    How much candy was taken in one grab: the box before and after (BGR images, the same crop of the same
    camera) and, if seen, the hand leaving. Returns (choice, confidence), choice one of the CHOICES values
    or "refused". Without a "before" photo it falls back to counting the hand alone.
    Network and API errors are raised to the caller.
    """
    if before is None or after is None:
        if hand is None:
            return "unclear", 0.0
        return count_candy(hand)
    images = [before, after] + ([hand] if hand is not None and hand.size else [])
    content = [{"type": "input_text", "text": "Image 1: the box before. Image 2: the box after."
                + (" Image 3: the hand leaving." if len(images) == 3 else "")}]
    content += [_image_part(image) for image in images]
    return _ask(content, TAKEN_QUESTION)


def count_candy(image):
    """
    Ask the Decisions API how much candy is in the hand in this BGR image.
    Returns (choice, confidence), where choice is one of the COUNT_QUESTION values,
    or "refused". Network and API errors are raised to the caller.
    """
    return _ask([_image_part(image)], COUNT_QUESTION)


if __name__ == "__main__":
    paths = sys.argv[1:]
    if len(paths) not in (1, 2, 3):
        raise SystemExit("Usage: python candy_counter.py before.jpg after.jpg [hand.jpg]  |  hand.jpg")
    images = [cv2.imread(path) for path in paths]
    for path, image in zip(paths, images):
        if image is None:
            raise SystemExit(f"Could not read {path}")
    if len(images) == 1:
        choice, confidence = count_candy(images[0])
    else:
        print(f"Box changed: {changed_fraction(images[0], images[1]):.1%} of it")
        choice, confidence = count_taken(*images)
    print(f"Candy taken: {choice} (confidence {confidence:.2f})")
