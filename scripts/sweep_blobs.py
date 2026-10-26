"""Sweep the blob detector on the TUNE half, score the held-out TEST half once."""
import glob, json, itertools, os, sys, statistics as st
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labvision.colonies import analyse_blobs

items = []
for j in sorted(glob.glob("data/agar_demo/**/*.json", recursive=True)):
    d = json.load(open(j)); gt = sum(1 for l in d["labels"] if l["class"] != "Contamination")
    if gt: items.append((j, gt))
tune, test = items[0::2], items[1::2]
imgs = {j: cv2.imread(j[:-4] + "jpg") for j, _ in items}
def run(rows, **kw):
    return [(gt, analyse_blobs(imgs[j], draw=False, **kw).count) for j, gt in rows]
def rel(res): return st.mean(abs(n - g) / g for g, n in res)
grid = dict(k=[8.0, 12.0, 16.0, 24.0, 40.0], margin=[0.80, 0.86, 0.92], sigmas=[(2.0, 3.0, 4.5), (1.5, 2.5, 4.0)])
best = None
for vals in itertools.product(*grid.values()):
    kw = dict(zip(grid, vals)); s = rel(run(tune, **kw)); print(kw, round(s, 3), flush=True)
    if best is None or s < best[0]: best = (s, kw)
print("BEST on tune:", best)
kw = best[1]
for name, rows in (("tune", tune), ("TEST (held out)", test)):
    res = run(rows, **kw)
    mae = st.mean(abs(n - g) for g, n in res)
    w20 = sum(abs(n - g) <= max(2, .2 * g) for g, n in res)
    print(f"{name:16s} n={len(res)} MAE={mae:.1f} mean-rel-err={rel(res):.2f} within20%={w20}/{len(res)}")
json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()}, open("labvision/params_blobs.json", "w"))
