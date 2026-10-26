"""Evaluate the agent on the AGAR demo plates. Writes out/eval_agent.json.

A. CLEAN: outcome per plate; error of ESTIMATE-outcome counts vs flagged ones (does the
   agent route hard plates to a human?).
B. REPAIR VALUE: same plates with repairs on vs off (the lower-res set is dim).
C. STRESS: real plates deliberately damaged (blur / dimming / glare patches);
   did the agent notice, did it call the right tool, did repair help the count?
Ground truth for counts = labelled colonies that are not 'Contamination'.
"""
import glob, json, os, sys, statistics as st
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labvision.agent import review

items = []
for j in sorted(glob.glob("data/agar_demo/**/*.json", recursive=True)):
    d = json.load(open(j)); gt = sum(1 for l in d["labels"] if l["class"] != "Contamination")
    sub = "lower" if "lower" in j else os.path.basename(os.path.dirname(j))
    items.append((j, gt, sub))

def rel(n, g): return abs(n - g) / g if g else None
out = {"clean": [], "repair": [], "stress": []}
def tools_used(r): return [t.get("tool") for t in r.trace if t.get("tool")]

for j, gt, sub in items:
    im = cv2.imread(j[:-4] + "jpg")
    r = review(im)
    out["clean"].append(dict(id=os.path.basename(j)[:-5], sub=sub, gt=gt, outcome=r.outcome, count=r.count,
                             rel=rel(r.count, gt) if r.count is not None else None, tools=tools_used(r), reason=r.reason))
    if sub == "lower" and gt:
        r0 = review(im, repairs=False)
        out["repair"].append(dict(id=os.path.basename(j)[:-5], gt=gt, with_repair=r.count, without=r0.count,
                                  out_with=r.outcome, out_without=r0.outcome))
    print("clean", sub, os.path.basename(j), r.outcome, r.count, gt, flush=True)

def blur(im): return cv2.GaussianBlur(im, (0, 0), 0.012 * max(im.shape[:2]))
def dim(im): return np.clip(im.astype(np.float32) * 0.40, 0, 255).astype(np.uint8)
def glare(im):
    """Five soft white highlights; only the highlight mask is feathered, the plate stays sharp."""
    h, w = im.shape[:2]; rng = np.random.default_rng(7)
    m = np.zeros((h, w), np.float32)
    for _ in range(5):
        c = (int(w * (0.3 + 0.4 * rng.random())), int(h * (0.3 + 0.4 * rng.random())))
        cv2.ellipse(m, c, (int(w * 0.07), int(h * 0.05)), int(rng.integers(0, 180)), 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), max(2.0, 0.003 * max(h, w)))[..., None]
    return np.clip(im.astype(np.float32) * (1 - m) + 255.0 * m, 0, 255).astype(np.uint8)

for j, gt, sub in items[1::2]:           # held-out half, to keep run time bounded
    if not gt: continue
    im = cv2.imread(j[:-4] + "jpg")
    for name, fn, want in (("blur", blur, "RETAKE"), ("dim", dim, "enhance"), ("glare", glare, "suppress_glare")):
        d = fn(im); r = review(d); r0 = review(d, repairs=False)
        ok = (r.outcome == "RETAKE") if want == "RETAKE" else (want in tools_used(r))
        out["stress"].append(dict(id=os.path.basename(j)[:-5], kind=name, gt=gt, outcome=r.outcome, count=r.count,
                                  noticed=ok, tools=tools_used(r), without=r0.count, out_without=r0.outcome))
    print("stress", os.path.basename(j), flush=True)
json.dump(out, open("out/eval_agent.json", "w"), indent=1)

print("\n=== A. CLEAN ===")
c = out["clean"]; acc = [x for x in c if x["outcome"] == "ESTIMATE" and x["rel"] is not None]
oth = [x for x in c if x["outcome"] != "ESTIMATE"]
from collections import Counter; print("outcomes:", dict(Counter(x["outcome"] for x in c)))
allrel = [x["rel"] for x in c if x["rel"] is not None]
print(f"all counted plates n={len(allrel)} median rel err {st.median(allrel):.2f}")
if acc: print(f"ESTIMATE-outcome n={len(acc)} median rel err {st.median(x['rel'] for x in acc):.2f} within20% {sum(x['rel']<=.2 for x in acc)}/{len(acc)}")
flag_counted = [x for x in oth if x["rel"] is not None]
if flag_counted: print(f"flagged-with-count n={len(flag_counted)} median rel err {st.median(x['rel'] for x in flag_counted):.2f}")
print("\n=== B. REPAIR VALUE (lower-res, dim set) ===")
for x in out["repair"]: print(x)
print("\n=== C. STRESS ===")
for kind in ("blur", "dim", "glare"):
    v = [x for x in out["stress"] if x["kind"] == kind]
    print(kind, f"noticed+right action {sum(x['noticed'] for x in v)}/{len(v)}")
