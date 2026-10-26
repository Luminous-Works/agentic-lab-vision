# Agentic Lab Vision: technical report

Luminous.Works LLC. PI: Tyrone Cunningham. OpenCV AI Competition 2026, Agentic Vision path.
Code: this repository. Live endpoint and video: see the Devpost submission.

## 1. Problem and users

Bacterial culture work still depends on a person looking at plates: is the photo usable,
how many colonies, does anything look wrong. That is slow, tedious and inconsistent.
The users we have in mind are small microbiology and phage-engineering labs that
photograph plates at the bench (the PI runs a Listeria phage-engineering project,
Project Phage, which is the motivation).

This entry is a **triage assistant for plate photos**, not a diagnostic. It decides
whether a photo can be trusted, repairs what it can, and gives a person an *estimate* to
approve. We say clearly below how accurate the count is, because it is not accurate
enough to run unattended.

## 2. What the system does

A photo goes to a web endpoint. An agent runs a perceive / decide / act loop
(`docs/agent_workflow.svg`):

1. **Perceive** (OpenCV 5): locate the dish; measure focus (contrast-normalised Laplacian
   variance), brightness, glare (fraction of near-saturated pixels), and count candidate
   colonies.
2. **Decide**: rules over those measurements pick the next action.
3. **Act**: call a repair tool and perceive again, stop and ask for a retake, escalate
   to a human, or report an estimate for approval.

What the images show changes what the system does next:

| OpenCV 5 observation | Next action |
|---|---|
| glare fraction above 0.005 | call `suppress_glare` (inpaint); the repaired pixels are **excluded from the count** and the hidden fraction is reported |
| mean brightness below 100 | call `enhance` (gamma + CLAHE), perceive again |
| dish not located | stop; ask for a retake; alert (SNS) |
| contrast-normalised sharpness below 0.04 | stop; ask for a retake; alert (SNS) |
| crowded plate, or more than 15% of the dish hidden | report the count but require human review; alert (SNS) |
| otherwise | report an **ESTIMATE**; a person approves or asks for a retake |

No count is ever recorded automatically. The page offers Approve / Retake and the choice
is logged (`POST /decision`). Every step, with the measurements behind it, is returned as
a trace. Real traces from the deployed endpoint are in `docs/live_traces.json`
(clean photo; dim photo, where `enhance` was called; glare photo, where `suppress_glare`
was called and 10% of the dish was excluded; blurry photo, which was refused).

The agent is rule-based. There is no language model at runtime, so every decision can be
read directly against the thresholds in `labvision/agent.py`.

## 3. OpenCV 5 implementation

OpenCV 5.0.0 (`opencv-python-headless==5.0.0.93`) is used for everything pixel-related
(`labvision/colonies.py`, `labvision/agent.py`):

- dish finding: `HoughCircles` at three sensitivities, then largest near-circular
  contour (`findContours`, `minEnclosingCircle`) in either polarity; if both fail the
  dish is reported as **not found** instead of guessing a centre;
- illumination flattening (`GaussianBlur` background subtraction), Otsu threshold,
  morphology, distance transform and `watershed` to split touching colonies;
- quality metrics: `Laplacian` (CV_32F in OpenCV 5; CV_64F on float32 input is not
  implemented there), grey statistics, saturation fraction;
- repair tools: `inpaint` (Telea) for glare, Lab conversion + `createCLAHE` for exposure;
- an overlay image (hidden regions tinted, colonies circled) so a person can check the
  count at a glance.

Two things were tried and are in the repository but **not used in production**: a
scale-space blob detector, and a learned filter over generous candidates
(`labvision/learned.py`; a random forest trained on the expert labels and exported as
plain arrays, because the OpenCV 5 headless build no longer ships `cv2.ml`). Neither beat
the simple counter on the held-out plates (section 5).

COOL (Cloud-Optimized OpenCV Library) is **not used**; this entry does not claim the COOL
award.

## 4. AWS deployment

(`docs/architecture.svg`, `deploy/`)

- AWS Lambda, Python 3.13, x86_64, 512 MB (the account's cap), public Function URL.
- OpenCV 5 plus numpy do not fit one 50 MB direct upload, so they are published as two
  layers (`lv-cv2`: the OpenCV engine; `lv-libs_numpy`: OpenCV's support libraries and
  numpy) by `deploy/build_layers.py`. `deploy/deploy.py` is idempotent and creates the
  SNS topic, the function and the URL (with both permission statements AWS now requires).
- Amazon SNS topic `lab-vision-alerts` receives RETAKE / NEEDS_HUMAN alerts.
- CloudWatch Logs holds one JSON line per verdict and per human decision.
- Security: requests need an `x-api-key` (constant-time comparison); payload capped at
  5 MB; the execution role is the AWS-managed Lambda logging policy plus SNS. The page
  resizes photos in the browser before upload. Reserved concurrency could not be set
  (account quota), so cost is bounded by the free tier and a billing budget instead.
- Observability: structured logs, the per-request trace, and `ms` (compute time) in
  every response.
- Reproducibility: pinned dependencies (`requirements.lock.txt`), scripted deploy,
  README with build and test steps. No servers, no database, no S3.

Measured on the live endpoint: 2 to 7 seconds of compute per 1600 px photo at 512 MB
(more when the repair loop runs), plus a cold start of a few seconds.

## 5. Evaluation (all numbers measured, nothing heard or eyeballed)

**Data.** The AGAR demo archive (Majchrowska et al., arXiv:2108.01234; CC BY-NC 2.0):
40 plates (three lighting groups at high resolution plus one low-resolution group), expert
colony labels and species. Ground truth for a count is the number of labelled colonies
that are not the `Contamination` class. 39 plates are countable. **This is a small sample
from one archive; no Listeria plates were available, so nothing here is evidence about
Listeria.**

**Colony counting is weak.** On the 38 plates where the agent produced a count:
mean absolute error 28.7 colonies, median relative error 0.64, within 20% (or 2
colonies) on 6 of 38. By lighting group the median relative errors are bright 0.53,
dark 0.64, vague 0.83, low-resolution 0.52. Tuning was done on half the plates and
scored once on the other half (19 plates): MAE 28.3, 5 of 19 within 20%. The learned
detector scored mean relative error 0.53 against about 1.0 for the simple counter on the
same split, but MAE 31.4 and 4 of 19 within 20%, so it was not adopted. Crowded plates
(100 to 300 colonies) are undercounted several-fold by every method.

**Photo triage works as designed.** Clean plates: 38 estimates, 1 sent to a human (the
one plate with no labelled colonies, flagged as unreliable), 1 retake (dish not found).
Stress test: each held-out plate (20) was deliberately damaged three ways and the agent's
response scored:

| Damage | Noticed and acted correctly | Outcome |
|---|---|---|
| heavy blur | 20 / 20 | all sent back for a retake, none counted |
| dimmed to 40% | 18 / 20 | 18 enhanced then counted; 2 retakes |
| five glare patches (about 10% of the dish) | 20 / 20 | 18 counted; 2 retakes |

Repair value, measured against the same agent with repairs switched off: glare repair
let the agent count 18 plates instead of 13, and on the 13 counted either way the median
count error was 0.71 with repair against 0.98 without (better on 10 of 13); with glare
the error on the counted plates (0.72) matches the same plates' clean error (0.72).
Dim repair changed median error from 0.63 to 0.57 (better on 10 of 18): a small effect.

**Contamination detection was not evaluated.** Only one demo plate carries
contamination labels (24 colonies), too few to measure anything.

## 6. Failure cases and limitations

- The counter often circles printed text on the dish as colonies and misses faint
  colonies (see `docs/live_overlay_3_glare.jpg`); touching colonies on crowded plates are
  merged. This is why every count is an estimate for a person to approve.
- One demo plate (525) never has its dish found; the agent correctly refuses it.
- Thresholds were set from this 40-plate sample and the stress test uses synthetic damage
  (blur, dimming, glare patches) applied to real plates; real bench problems will differ.
- Not built, although the original proposal described them: camera-based optical-density
  estimation (needs a calibrated optical bench), live multi-camera monitoring on an edge
  device, and any Listeria-specific classification. No claim is made about them.
- The agent never learns; thresholds are fixed rules.

## 7. Responsible use

- It is a triage aid for trained staff. It must not replace confirmatory methods or be
  used to release or reject anything on its own; the interface says "ESTIMATE" and
  requires approval.
- Misleading confidence is the main risk. The system therefore states its measured accuracy,
  shows an overlay for checking, escalates uncertain photos, and logs human decisions.
- Data: only the public AGAR demo images are used, under their licence (non-commercial,
  credited). The deployed service stores no photos; logs hold verdicts, not images.
- Disclosure: the code was written with an AI coding assistant (Claude Code). The deployed
  agent contains no language model.
