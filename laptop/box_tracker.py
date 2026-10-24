"""
Find the candy box by the ArUco markers taped to its corners, so it can be moved without redrawing anything.

Print the markers once:  python box_tracker.py
  This writes markers.png: four markers, IDs 0 to 3. Print it, cut them out and tape one on each corner of
  the box's opening, facing the camera, in order around the rim (0, 1, 2, 3 clockwise as the camera sees
  them; any corner can be 0).

hand_in_bowl.py calls BoxTracker.update() as it goes. A hand reaching in covers markers, so each marker
keeps the last place it was seen: the box only moves when someone moves it, and then they're seen again.
Three markers are enough to find the box (the fourth corner is filled in), or two on opposite corners.
"""
import cv2
import numpy as np

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MARKER_IDS = (0, 1, 2, 3)  # in order around the rim
MARGIN = 0.12              # grow the outline by this fraction round its centre, so a hand over the rim counts
JITTER_PIXELS = 4          # a marker has to move further than this to count as moved


class BoxTracker:
    def __init__(self):
        self._detector = cv2.aruco.ArucoDetector(DICTIONARY, cv2.aruco.DetectorParameters())
        self.corners = {}  # marker ID -> (x, y) of its centre as 0-1 fractions of the frame, as last seen
        self.visible = 0   # markers seen in the latest update

    def update(self, frame):
        """Looks for the markers in this frame. Returns the box outline (see outline())."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, ids, _ = self._detector.detectMarkers(gray)
        fh, fw = frame.shape[:2]
        self.visible = 0
        if ids is not None:
            for marker, marker_id in zip(found, ids.flatten()):
                marker_id = int(marker_id)
                if marker_id not in MARKER_IDS:
                    continue
                self.visible += 1
                cx, cy = marker[0].mean(axis=0)
                old = self.corners.get(marker_id)
                if old is None or abs(cx - old[0] * fw) > JITTER_PIXELS or abs(cy - old[1] * fh) > JITTER_PIXELS:
                    self.corners[marker_id] = (cx / fw, cy / fh)
        return self.outline()

    def forget(self):
        """Drop every remembered marker (the box was moved while some were covered)."""
        self.corners.clear()

    def outline(self):
        """
        The box's opening as an array of (x, y) points, 0-1 fractions of the frame, grown by MARGIN.
        None until enough markers have been seen.
        """
        known = {i: np.array(p, np.float32) for i, p in self.corners.items()}
        if len(known) == 3:
            # A parallelogram's missing corner: the two corners beside it, minus the one opposite.
            (missing,) = set(MARKER_IDS) - set(known)
            before, after, opposite = (missing - 1) % 4, (missing + 1) % 4, (missing + 2) % 4
            known[missing] = known[before] + known[after] - known[opposite]
        elif len(known) == 2:
            a, b = sorted(known)
            if b - a != 2:
                return None  # two neighbouring corners are only one edge of the box
            (x1, y1), (x2, y2) = np.minimum(known[a], known[b]), np.maximum(known[a], known[b])
            known = {0: np.array([x1, y1]), 1: np.array([x2, y1]), 2: np.array([x2, y2]), 3: np.array([x1, y2])}
        elif len(known) < 2:
            return None
        points = np.array([known[i] for i in sorted(known)], np.float32)
        centre = points.mean(axis=0)
        return np.clip(centre + (points - centre) * (1 + MARGIN), 0, 1).astype(np.float32)


def grow(outline, factor):
    """The outline scaled by `factor` round its centre (1.5 = half as big again)."""
    centre = outline.mean(axis=0)
    return np.clip(centre + (outline - centre) * factor, 0, 1).astype(np.float32)


def contains(outline, x, y):
    """True if the point (0-1 fractions of the frame) is inside the outline."""
    return cv2.pointPolygonTest(outline.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0


def crop(frame, outline):
    """The part of the frame the outline covers (its bounding rectangle), as a copy."""
    fh, fw = frame.shape[:2]
    x1, y1 = (outline.min(axis=0) * (fw, fh)).astype(int)
    x2, y2 = (outline.max(axis=0) * (fw, fh)).astype(int)
    return frame[max(y1, 0):min(y2, fh), max(x1, 0):min(x2, fw)].copy()


def draw(frame, outline, color, thickness=3):
    fh, fw = frame.shape[:2]
    points = (outline * (fw, fh)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [points], True, color, thickness)


if __name__ == "__main__":
    # A sheet of the four markers with a white border (the detector needs it), labelled.
    size, border = 400, 60
    tiles = []
    for marker_id in MARKER_IDS:
        marker = cv2.aruco.generateImageMarker(DICTIONARY, marker_id, size)
        tile = cv2.copyMakeBorder(marker, border, border + 40, border, border, cv2.BORDER_CONSTANT, value=255)
        cv2.putText(tile, f"ID {marker_id}", (border, size + border + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
        tiles.append(tile)
    sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite("markers.png", sheet)
    print("Wrote markers.png. Print it, cut out the four markers and tape them to the box's corners,")
    print("0, 1, 2, 3 clockwise as the camera sees them.")
