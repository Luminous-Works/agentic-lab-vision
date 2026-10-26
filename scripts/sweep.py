"""Parameter sweep on a TUNE half; the held-out TEST half is scored once at the end.

Split: images sorted by (subset, id); even positions tune, odd positions test.
Objective: mean relative count error (so crowded plates do not dominate).
"""
import glob, json, itertools, os, sys, statistics as st
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labvision.colonies import analyse

items = []
for j in sorted(glob.glob("data/agar_demo/**/*.json", recursive=True)):
    d = json.load(open(j))
    gt = sum(1 for l in d["labels"] if l["class"] != "Contamination")
    if gt:
        items.append((j, gt))
tune, test = items[0::2], items[1::2]
cache = {}
def img(j):
    if j not in cache:
        im = cv2.imread(j[:-4] + "jpg"); f = 1000 / max(im.shape[:2])
        cache[j] = im  # analyse() rescales itself
    return cache[j]

def score(rows, **kw):
    errs = [abs(analyse(img(j), draw=False, **kw).count - gt) / gt for j, gt in rows]
    return st.mean(errs)

grid = dict(tmul=[0.4, 0.6, 0.8, 1.0], tmin=[8, 12, 18], peak=[0.35, 0.5], bgdiv=[3.0, 6.0])
best = None
for vals in itertools.product(*grid.values()):
    kw = dict(zip(grid, vals))
    s = score(tune, **kw)
    print(kw, round(s, 3), flush=True)
    if best is None or s < best[0]:
        best = (s, kw)
print("BEST on tune:", best)
kw = best[1]
for name, rows in (("tune", tune), ("TEST (held out)", test)):
    res = [(gt, analyse(img(j), draw=False, **kw).count) for j, gt in rows]
    mae = st.mean(abs(n - g) for g, n in res); rel = st.mean(abs(n - g) / g for g, n in res)
    w20 = sum(abs(n - g) <= max(2, .2 * g) for g, n in res)
    print(f"{name:16s} n={len(res)} MAE={mae:.1f} mean-rel-err={rel:.2f} within20%={w20}/{len(res)}")
json.dump(kw, open("labvision/params.json", "w"))
