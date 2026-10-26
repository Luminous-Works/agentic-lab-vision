"""Perception -> decision -> action loop for plate review (Agentic Lab Vision).

The agent never trusts a count blindly. It first perceives the *quality* of the
photo with OpenCV 5, and what it sees decides the next tool call:

  dish not found / photo too blurry  -> stop, ask a human to retake (no count made)
  too dark                           -> call enhance (gamma + CLAHE), perceive again
  glare                              -> call suppress_glare (inpaint), perceive again
  too crowded for a trustworthy count-> count, but ask a human to approve the number
  otherwise                          -> count and accept

Every step lands in `trace`, so the visual evidence behind each decision is
inspectable. Thresholds are in THRESH and were set from the AGAR demo statistics
(see scripts/profile_quality.py), not guessed.

The agent is deliberately rule-based: no model is called, so it is deterministic,
free to run, and its failure modes are the rules' failure modes (listed in the
report). Human control: NEEDS_HUMAN outcomes carry a reason and never auto-accept.
"""
from dataclasses import dataclass, field
import json, os
import cv2
import numpy as np
from .colonies import analyse

_P = os.path.join(os.path.dirname(__file__), "params.json")
PARAMS = json.load(open(_P)) if os.path.exists(_P) else {}

THRESH = dict(
    sharp_min=25.0,      # variance of Laplacian (1000 px working size) below this = too blurry
    bright_min=80.0,     # mean grey inside the dish below this = under-exposed
    glare_max=0.02,      # fraction of dish pixels near saturation above this = glare
    crowded_cov=0.25,    # fraction of dish area covered by blobs above this
    crowded_n=250,       # or more blobs than this: count unreliable, ask a human
    max_repairs=2,
)


# ---------------------------------------------------------------- tools
def tool_enhance(img):
    """Lift an under-exposed photo: gamma then CLAHE on lightness."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l = lab[..., 0].astype(np.float32) / 255.0
    mean = max(float(l.mean()), 1e-3)
    gamma = float(np.clip(np.log(0.5) / np.log(mean), 0.4, 1.0))
    l = np.power(l, gamma)
    lab[..., 0] = np.clip(l * 255, 0, 255).astype(np.uint8)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), {"gamma": round(gamma, 2)}


def tool_suppress_glare(img):
    """Inpaint near-saturated pixels (specular highlights)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = (g > 250).astype(np.uint8) * 255
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return cv2.inpaint(img, m, 5, cv2.INPAINT_TELEA), {"masked_px": int((m > 0).sum())}


TOOLS = {"enhance": tool_enhance, "suppress_glare": tool_suppress_glare}


# ---------------------------------------------------------------- perception
def perceive(img):
    a = analyse(img, draw=False, **PARAMS)
    return a, dict(dish_found=a.dish_found, sharpness=round(a.sharpness, 1),
                   brightness=round(a.brightness, 1), glare=round(a.glare, 4),
                   coverage=round(a.coverage, 3), count=a.count)


@dataclass
class Result:
    outcome: str                 # ACCEPT | NEEDS_HUMAN | RETAKE
    count: int | None
    reason: str
    trace: list = field(default_factory=list)
    alert: bool = False

    def to_dict(self):
        return dict(outcome=self.outcome, count=self.count, reason=self.reason,
                    alert=self.alert, trace=self.trace)


def review(img, notify=None, repairs=True):
    """Run the loop on one BGR image. `repairs=False` disables the enhance/glare tools
    (used by the evaluation to measure what the repairs are worth)."""
    T = THRESH
    trace = []
    cur = img
    for step in range(T["max_repairs"] + 1):
        a, obs = perceive(cur)
        entry = {"step": step, "observation": obs}
        # --- decision on the visual evidence
        if not obs["dish_found"]:
            entry["decision"] = "dish not located -> stop, ask human to retake"
            trace.append(entry)
            r = Result("RETAKE", None, "dish not found in the photo", trace, alert=True)
            break
        if obs["sharpness"] < T["sharp_min"]:
            entry["decision"] = "photo too blurry to count -> stop, ask human to retake"
            trace.append(entry)
            r = Result("RETAKE", None, "photo out of focus", trace, alert=True)
            break
        if repairs and obs["glare"] > T["glare_max"] and step < T["max_repairs"]:
            cur, info = TOOLS["suppress_glare"](cur)
            entry.update(decision="glare high -> call suppress_glare, perceive again", tool="suppress_glare", tool_result=info)
            trace.append(entry); continue
        if repairs and obs["brightness"] < T["bright_min"] and step < T["max_repairs"]:
            cur, info = TOOLS["enhance"](cur)
            entry.update(decision="under-exposed -> call enhance, perceive again", tool="enhance", tool_result=info)
            trace.append(entry); continue
        if obs["coverage"] > T["crowded_cov"] or obs["count"] > T["crowded_n"]:
            entry["decision"] = "too crowded for a trustworthy count -> report it, ask human to approve"
            trace.append(entry)
            r = Result("NEEDS_HUMAN", obs["count"], "crowded plate: count likely unreliable", trace, alert=True)
            break
        entry["decision"] = "quality adequate -> accept count"
        trace.append(entry)
        r = Result("ACCEPT", obs["count"], "quality checks passed", trace)
        break
    else:  # repairs exhausted without a verdict
        r = Result("NEEDS_HUMAN", None, "could not repair photo within the step budget", trace, alert=True)
    if r.alert and notify:
        notify(f"[Agentic Lab Vision] {r.outcome}: {r.reason}")
    return r
