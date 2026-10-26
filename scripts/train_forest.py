"""Train the colony-vs-not forest on the AGAR demo labels and export it as arrays.

Split BY IMAGE (never by candidate): even-indexed plates train, odd-indexed plates
are the held-out test. The keep-threshold is chosen on the TRAIN plates only
(minimum mean relative count error), then the test plates are scored once.
Needs scikit-learn (training machine only; see requirements-train.txt).
Run from the project root:  python scripts/train_forest.py
"""
import glob
import json
import os
import statistics as st
import sys

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labvision.learned import SIGMAS, candidates, merge  # noqa: E402

items = []
for j in sorted(glob.glob("data/agar_demo/**/*.json", recursive=True)):
    d = json.load(open(j))
    gt = sum(1 for l in d["labels"] if l["class"] != "Contamination")
    if gt:
        items.append((j, d))
train_items, test_items = items[0::2], items[1::2]


def prep(item):
    j, d = item
    pts, X, ctx = candidates(cv2.imread(j[:-4] + "jpg"))
    s = ctx["scale"]
    cen = np.array([[(l["x"] + l["width"] / 2) * s, (l["y"] + l["height"] / 2) * s] for l in d["labels"]
                    if l["class"] != "Contamination"], np.float32)
    rad = np.array([max(4.0, 0.5 * min(l["width"], l["height"]) * s) for l in d["labels"]
                    if l["class"] != "Contamination"], np.float32)
    y = np.zeros(len(pts), np.int32)
    if len(pts) and len(cen):
        dist = np.hypot(pts[:, None, 0] - cen[None, :, 0], pts[:, None, 1] - cen[None, :, 1])
        y = (dist <= rad[None, :] * 1.2).any(1).astype(np.int32)
    sig = np.array(SIGMAS, np.float32)[ctx["bidx"]] if len(pts) else np.zeros(0, np.float32)
    return dict(pts=pts, X=X, y=y, sig=sig, gt=len(cen), id=os.path.basename(j)[:-5])


print("building candidates...", flush=True)
tr = [prep(i) for i in train_items]
te = [prep(i) for i in test_items]
for p in tr[:3]:
    print("train", p["id"], "cands", len(p["pts"]), "pos", int(p["y"].sum()), "gt", p["gt"])
Xtr = np.vstack([p["X"] for p in tr])
ytr = np.concatenate([p["y"] for p in tr])
print("train rows", Xtr.shape, "positives", int(ytr.sum()), flush=True)
rf = RandomForestClassifier(n_estimators=60, max_depth=8, min_samples_leaf=8, class_weight="balanced",
                            n_jobs=-1, random_state=0).fit(Xtr, ytr)


def counts(plates, thr):
    out = []
    for p in plates:
        if len(p["pts"]) == 0:
            out.append((p["gt"], 0))
            continue
        prob = rf.predict_proba(p["X"])[:, 1]
        out.append((p["gt"], len(merge(p["pts"], prob, p["sig"], thr))))
    return out


rel = lambda res: st.mean(abs(n - g) / g for g, n in res)
best = min(((rel(counts(tr, t)), t) for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)))
thr = best[1]
print("chosen threshold on TRAIN plates:", thr, "train mean-rel-err %.2f" % best[0])
for name, plates in (("train", tr), ("TEST (held out)", te)):
    res = counts(plates, thr)
    mae = st.mean(abs(n - g) for g, n in res)
    w20 = sum(abs(n - g) <= max(2, .2 * g) for g, n in res)
    print(f"{name:16s} n={len(res)} MAE={mae:.1f} mean-rel-err={rel(res):.2f} within20%={w20}/{len(res)}")
print("held-out detail (gt, pred):", counts(te, thr))


# soft count: expected number of colonies = sum of probabilities, one global scale
# factor fitted on the TRAIN plates only
def soft(plates):
    return [(p["gt"], float(rf.predict_proba(p["X"])[:, 1].sum()) if len(p["pts"]) else 0.0) for p in plates]


s_tr, s_te = soft(tr), soft(te)
factor = float(np.median([g / max(s, 1e-6) for g, s in s_tr]))
print("soft-count scale factor (train):", round(factor, 3))
for name, v in (("train", s_tr), ("TEST (held out)", s_te)):
    res = [(g, round(factor * s)) for g, s in v]
    mae = st.mean(abs(n - g) for g, n in res)
    w20 = sum(abs(n - g) <= max(2, .2 * g) for g, n in res)
    a = np.array([g for g, _ in res], float)
    b = np.array([n for _, n in res], float)
    r2 = 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    print(f"SOFT {name:16s} n={len(res)} MAE={mae:.1f} mean-rel-err={rel(res):.2f} within20%={w20}/{len(res)} R2={r2:.2f} corr={np.corrcoef(a, b)[0, 1]:.2f}")
print("soft held-out detail (gt, pred):", [(g, round(factor * s)) for g, s in s_te])

# ---- export as arrays
T = len(rf.estimators_)
M = max(e.tree_.node_count for e in rf.estimators_)
feat = -np.ones((T, M), np.int32)
thrs = np.zeros((T, M), np.float32)
left = np.zeros((T, M), np.int32)
right = np.zeros((T, M), np.int32)
val = np.zeros((T, M), np.float32)
for t, e in enumerate(rf.estimators_):
    tr_ = e.tree_
    n = tr_.node_count
    feat[t, :n] = np.where(tr_.children_left == -1, -1, tr_.feature)
    thrs[t, :n] = tr_.threshold
    left[t, :n] = np.where(tr_.children_left == -1, 0, tr_.children_left)
    right[t, :n] = np.where(tr_.children_right == -1, 0, tr_.children_right)
    v = tr_.value[:, 0, :]
    val[t, :n] = v[:, 1] / np.maximum(v.sum(1), 1e-9)
np.savez_compressed("labvision/forest.npz", feat=feat, thr=thrs, left=left, right=right, val=val,
                    threshold=np.float32(thr))
print("saved labvision/forest.npz", os.path.getsize("labvision/forest.npz") // 1024, "KB")
