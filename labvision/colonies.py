"""Colony analysis on a petri-dish photo with OpenCV 5.

Pure image processing, no learned weights. Steps:
  1. downscale so the dish is ~1000 px across (speed, resolution independence)
  2. find the dish (Hough circle, falls back to the central disc) and mask to
     its inside, shrunk by a margin so the rim and the glare are excluded
  3. flatten the lighting: subtract a heavy blur (illumination estimate)
  4. threshold the flattened image, clean with morphology
  5. split touching colonies with a distance-transform watershed
  6. return the colony blobs (centroid, area, mean colour) and quality flags

Everything the agent needs to decide on is returned in the Analysis object.
"""
from dataclasses import dataclass, field
import cv2
import numpy as np

TARGET = 1000  # working long-side in px


@dataclass
class Colony:
    x: float
    y: float
    area: float
    bgr: tuple


@dataclass
class Analysis:
    scale: float
    dish: tuple            # (cx, cy, r) in working pixels
    colonies: list = field(default_factory=list)
    coverage: float = 0.0  # fraction of dish area covered by colonies
    sharpness: float = 0.0 # variance of Laplacian inside the dish
    glare: float = 0.0     # fraction of dish pixels near saturation
    brightness: float = 0.0
    overlay: np.ndarray = None

    @property
    def count(self):
        return len(self.colonies)


def _find_dish(gray):
    h, w = gray.shape
    small = cv2.GaussianBlur(gray, (0, 0), 3)
    m = min(h, w)
    circles = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp=1.5, minDist=m,
                               param1=100, param2=40,
                               minRadius=int(m * 0.30), maxRadius=int(m * 0.55))
    if circles is not None:
        cx, cy, r = circles[0][0]
        return float(cx), float(cy), float(r)
    return w / 2.0, h / 2.0, m * 0.45


def analyse(bgr, margin=0.90, min_area=12, max_area_frac=0.02, draw=True):
    h0, w0 = bgr.shape[:2]
    scale = TARGET / max(h0, w0)
    img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cx, cy, r = _find_dish(gray)

    mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(r * margin), 255, -1)
    area_dish = float(cv2.countNonZero(mask))

    # flatten illumination: heavy blur of the masked image is the background
    g = gray.astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), max(r / 6.0, 15))
    diff = np.abs(g - bg)
    diff[mask == 0] = 0
    diff8 = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inside = diff8[mask > 0]
    t, _ = cv2.threshold(inside.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = ((diff8 > max(t, 18)) & (mask > 0)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    # watershed split of touching colonies
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, peaks = cv2.threshold(dist, 0.45 * dist.max() if dist.max() > 0 else 1, 255, cv2.THRESH_BINARY)
    peaks = peaks.astype(np.uint8)
    n, markers = cv2.connectedComponents(peaks)
    markers = markers + 1
    unknown = cv2.subtract(binary, peaks)
    markers[unknown == 255] = 0
    markers = cv2.watershed(img.copy(), markers.astype(np.int32))

    colonies = []
    max_area = max_area_frac * area_dish
    for lab in range(2, markers.max() + 1):
        blob = (markers == lab)
        a = float(blob.sum())
        if a < min_area or a > max_area:
            continue
        ys, xs = np.nonzero(blob)
        b, gch, rch = (float(img[..., c][blob].mean()) for c in range(3))
        colonies.append(Colony(float(xs.mean()), float(ys.mean()), a, (b, gch, rch)))

    inner = gray[mask > 0]
    out = Analysis(scale=scale, dish=(cx, cy, r), colonies=colonies,
                   coverage=sum(c.area for c in colonies) / area_dish,
                   sharpness=float(cv2.Laplacian(gray, cv2.CV_64F)[mask > 0].var()),
                   glare=float((inner > 250).mean()),
                   brightness=float(inner.mean()))
    if draw:
        ov = img.copy()
        cv2.circle(ov, (int(cx), int(cy)), int(r * margin), (255, 200, 0), 2)
        for c in colonies:
            cv2.circle(ov, (int(c.x), int(c.y)), max(3, int((c.area / np.pi) ** 0.5) + 2), (0, 0, 255), 1)
        out.overlay = ov
    return out
