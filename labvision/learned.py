"""Learned colony detector: generous candidates (OpenCV 5) + a small random forest.

Why: threshold rules cannot tell a colony from a rim arc, a printed label or sensor
noise. So candidates are generated generously (difference-of-Gaussians maxima at
four scales), each is described by 19 local features (contrast at its own scale,
colour offset from the agar, texture, distance from the rim, candidate density,
plate-level context), and a forest trained on the expert labels keeps the real ones.

Runtime needs only OpenCV 5 + numpy: the trained forest is stored as plain arrays
(labvision/forest.npz) and evaluated here, so no ML library ships to the cloud.
Training lives in scripts/train_forest.py (scikit-learn, training machine only).
"""
import os

import cv2
import numpy as np

from .colonies import Analysis, Colony, TARGET, _find_dish

SIGMAS = (1.5, 2.5, 4.0, 6.0)
K_CAND = 1.8          # generous candidate threshold (MAD units); the forest does the filtering
MARGIN = 0.92
NMS_PX = 3
MAX_CAND = 9000
FOREST_PATH = os.path.join(os.path.dirname(__file__), "forest.npz")


def candidates(bgr, margin=MARGIN, k=K_CAND):
    """Return (points (N,2) xy, features (N,F), context dict) at the 1000 px working size."""
    h0, w0 = bgr.shape[:2]
    scale = TARGET / max(h0, w0)
    img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    cx, cy, r, found = _find_dish(gray.astype(np.uint8))
    mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(r * margin), 255, -1)
    inside = mask > 0
    g = lambda a, s: cv2.GaussianBlur(a, (0, 0), s)

    signed, absd = [], []
    for s in SIGMAS:
        d = (s * s) * (g(gray, s) - g(gray, 1.6 * s))
        signed.append(d)
        absd.append(np.abs(d))
    signed, absd = np.stack(signed), np.stack(absd)
    best = absd.max(0)
    bidx = absd.argmax(0)
    vals = best[inside]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1e-6
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * NMS_PX + 1, 2 * NMS_PX + 1))
    peaks = (best >= cv2.dilate(best, kern)) & (best > med + k * mad) & inside
    ys, xs = np.nonzero(peaks)
    if len(ys) > MAX_CAND:
        keep = np.argsort(-best[ys, xs])[:MAX_CAND]
        ys, xs = ys[keep], xs[keep]
    ctx = dict(scale=scale, dish=(cx, cy, r), dish_found=found, img=img, gray=gray, mask=mask,
               bidx=bidx[ys, xs], best=best[ys, xs])
    if len(ys) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 19), np.float32), ctx

    # context maps
    Lmed = float(np.median(lab[..., 0][inside]))
    amed = float(np.median(lab[..., 1][inside]))
    bmed = float(np.median(lab[..., 2][inside]))
    dens_img = np.zeros(gray.shape, np.float32)
    dens_img[ys, xs] = 1.0
    dens = cv2.boxFilter(dens_img, -1, (41, 41), normalize=False)
    feats = np.zeros((len(ys), 19), np.float32)
    for c, s in enumerate(SIGMAS):
        feats[:, c] = absd[c][ys, xs] / mad
        feats[:, 4 + c] = signed[c][ys, xs] / mad
    sb = np.array(SIGMAS, np.float32)[ctx["bidx"]]
    feats[:, 8] = sb
    # per-scale local means (computed once per scale, gathered by each point's best scale)
    gL, g3, gsq, ga, gb = {}, {}, {}, {}, {}
    for s in SIGMAS:
        gL[s], g3[s] = g(gray, s), g(gray, 3 * s)
        gsq[s] = g(gray * gray, 2 * s)
        ga[s], gb[s] = g(lab[..., 1], s), g(lab[..., 2], s)
    for i, (y, x, s) in enumerate(zip(ys, xs, sb)):
        s = float(s)
        feats[i, 9] = gL[s][y, x] - g3[s][y, x]
        m2 = g3[s][y, x]
        feats[i, 10] = np.sqrt(max(gsq[s][y, x] - m2 * m2, 0.0))
        feats[i, 11] = gL[s][y, x] - Lmed
        feats[i, 12] = ga[s][y, x] - amed
        feats[i, 13] = gb[s][y, x] - bmed
    feats[:, 14] = np.hypot(xs - cx, ys - cy) / r
    feats[:, 15] = ctx["best"] / mad
    feats[:, 16] = dens[ys, xs]
    feats[:, 17] = float(gray[inside].mean())
    feats[:, 18] = mad
    pts = np.stack([xs, ys], 1).astype(np.float32)
    ctx["mad"] = mad
    return pts, feats, ctx


class Forest:
    """Random forest stored as arrays; probability of 'colony' for each row."""
    def __init__(self, path=FOREST_PATH):
        z = np.load(path)
        self.feat, self.thr, self.left, self.right, self.val = (z[k] for k in ("feat", "thr", "left", "right", "val"))
        self.threshold = float(z["threshold"])

    def proba(self, X):
        n_trees = self.feat.shape[0]
        out = np.zeros(len(X), np.float32)
        for t in range(n_trees):
            node = np.zeros(len(X), np.int32)
            while True:
                f = self.feat[t][node]
                internal = f >= 0
                if not internal.any():
                    break
                go_left = X[np.arange(len(X)), np.where(internal, f, 0)] <= self.thr[t][node]
                nxt = np.where(go_left, self.left[t][node], self.right[t][node])
                node = np.where(internal, nxt, node)
            out += self.val[t][node]
        return out / n_trees


def merge(pts, prob, sig, keep_thr):
    """Greedy suppression: strongest first, drop others closer than ~1.8 sigma."""
    order = np.argsort(-prob)
    kept = []
    for i in order:
        if prob[i] < keep_thr:
            break
        if all(np.hypot(*(pts[i] - pts[j])) > 1.8 * max(sig[i], sig[j]) for j in kept):
            kept.append(i)
    return kept


def analyse_learned(bgr, forest=None, draw=True, margin=MARGIN):
    forest = forest or Forest()
    pts, X, ctx = candidates(bgr, margin=margin)
    cx, cy, r = ctx["dish"]
    gray, mask, img = ctx["gray"], ctx["mask"], ctx["img"]
    colonies = []
    if len(pts):
        prob = forest.proba(X)
        sig = np.array(SIGMAS, np.float32)[ctx["bidx"]]
        for i in merge(pts, prob, sig, forest.threshold):
            x, y = pts[i]
            b, gg, rr = (float(img[int(y), int(x), c]) for c in range(3))
            colonies.append(Colony(float(x), float(y), float(np.pi * 2 * sig[i] ** 2), (b, gg, rr)))
    inner = gray.astype(np.uint8)[mask > 0]
    area_dish = float(cv2.countNonZero(mask))
    out = Analysis(scale=ctx["scale"], dish=ctx["dish"], colonies=colonies,
                   coverage=sum(c.area for c in colonies) / area_dish,
                   sharpness=float(cv2.Laplacian(gray, cv2.CV_32F)[mask > 0].var()),
                   glare=float((inner > 250).mean()), brightness=float(inner.mean()),
                   dish_found=ctx["dish_found"])
    if draw:
        ov = img.copy()
        cv2.circle(ov, (int(cx), int(cy)), int(r * margin), (255, 200, 0), 2)
        for c in colonies:
            cv2.circle(ov, (int(c.x), int(c.y)), max(3, int((c.area / np.pi) ** 0.5) + 2), (0, 0, 255), 1)
        out.overlay = ov
    return out
