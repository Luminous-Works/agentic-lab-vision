"""Evaluate colony counts against the AGAR demo labels (40 images).

Ground truth = number of labelled colonies that are not 'Contamination'.
Prints per-subset MAE and the fraction within 10 % / 20 % (or +-2 colonies).
"""
import glob, json, os, sys, statistics as st
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labvision.colonies import analyse

root = "data/agar_demo"
rows = []
for j in sorted(glob.glob(f"{root}/**/*.json", recursive=True)):
    d = json.load(open(j))
    gt = sum(1 for l in d["labels"] if l["class"] != "Contamination")
    if gt == 0:
        continue
    sub = os.path.basename(os.path.dirname(j))
    if "lower" in j:
        sub = "lower-res"
    a = analyse(cv2.imread(j[:-4] + "jpg"), draw=False)
    rows.append((sub, os.path.basename(j), gt, a.count))
by = {}
for sub, name, gt, n in rows:
    by.setdefault(sub, []).append((gt, n))
    print(f"{sub:10s} {name:12s} gt={gt:4d} pred={n:4d} err={n-gt:+d}")
print()
def ok(gt, n, f): return abs(n - gt) <= max(2, f * gt)
tot = [(g, n) for v in by.values() for g, n in v]
for sub, v in list(by.items()) + [("ALL", tot)]:
    mae = st.mean(abs(n - g) for g, n in v)
    rel = st.mean(abs(n - g) / g for g, n in v)
    print(f"{sub:10s} n={len(v):2d} MAE={mae:6.1f}  mean rel err={rel:5.2f}  "
          f"within10%={sum(ok(g,n,.1) for g,n in v)}/{len(v)}  within20%={sum(ok(g,n,.2) for g,n in v)}/{len(v)}")
