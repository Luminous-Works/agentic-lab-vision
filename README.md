# Agentic Lab Vision

OpenCV AI Competition 2026 (powered by AWS) entry, Agentic Vision path.
Luminous.Works LLC, PI: Tyrone Cunningham.

A culture-plate photo goes in. OpenCV 5 checks the photo itself (is the dish there,
is it in focus, is it lit properly, is there glare), counts colonies, and a small
perception-decision-action loop decides what happens next: repair the photo and look
again, send it back for a retake, or hand a person an **estimate** to approve.
No count is ever recorded automatically.

Read `REPORT.md` for the problem, architecture, evaluation, failure cases and
limitations. Short version of the honest status: the photo-quality triage works
(blur 20/20, dim 18/20 in a stress test); the colony **counter is weak** (see the
numbers), which is why a human approves every count.

## What is in this repository

| Path | What it is |
|---|---|
| `labvision/colonies.py` | OpenCV 5 analysis: dish finder, illumination flattening, threshold, watershed split, quality metrics |
| `labvision/agent.py` | the perceive / decide / act loop, its tools (`enhance`, `suppress_glare`), thresholds, trace |
| `labvision/learned.py`, `scripts/train_forest.py` | an experiment: learned candidate filter (not used in production, see report) |
| `lambda_function.py`, `page.py` | the AWS Lambda handler and the review web page |
| `deploy/build_layers.py`, `deploy/deploy.py` | reproducible AWS deployment |
| `scripts/eval_counts.py`, `scripts/sweep.py`, `scripts/eval_agent.py` | evaluation scripts |
| `docs/architecture.svg`, `docs/agent_workflow.svg` | diagrams |
| `docs/live_traces.json` | traces captured from the live endpoint |
| `demo/` | four demo photos derived from one AGAR plate: clean, dim, glare, blurry |

## Requirements

Python 3.13. Runtime dependencies are pinned in `requirements.lock.txt`
(`opencv-python-headless==5.0.0.93`, `numpy==2.5.4`, plus `boto3`/`awscli` for deployment).
`requirements-train.txt` (scikit-learn) is only for the learned-detector experiment.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock.txt      # Windows; use .venv/bin on Linux
```

## Data

The AGAR demo archive (40 plates, expert labels), CC BY-NC 2.0, from
https://agar.neurosys.com/ . Download `AGAR_demo.zip`, unzip into `data/agar_demo/`
(`data/` is git-ignored). Credit: Majchrowska et al., "AGAR a microbial colony dataset
for deep learning detection", arXiv:2108.01234.

## Run the evaluations locally

```bash
.venv/Scripts/python scripts/eval_counts.py     # count accuracy per lighting subset
.venv/Scripts/python scripts/sweep.py           # tune on half the plates, score the other half once
.venv/Scripts/python scripts/eval_agent.py      # agent outcomes, repair value, stress test (about 12 min)
```

## Deploy to AWS

Needs an AWS profile named `labvision` (an IAM user with Lambda, SNS and CloudWatch Logs
access) and an execution role `lab-vision-lambda-role` that Lambda can assume
(`AWSLambdaBasicExecutionRole` and `AmazonSNSFullAccess`).

```bash
.venv/Scripts/python deploy/build_layers.py     # publishes lv-cv2 and lv-libs_numpy (OpenCV 5 split in two, 50 MB limit)
.venv/Scripts/python deploy/deploy.py           # SNS topic, function, public Function URL; token saved to out/deploy.json
```

`deploy.py` is idempotent. Everything stays inside the Lambda / SNS / CloudWatch free
tier at demo volumes. No servers, databases or S3 are used.

## Use it

Open the Function URL in a browser, enter the API key, choose a plate photo, press
Analyse. Or from a shell:

```bash
curl -s -X POST "$URL/analyze" -H "x-api-key: $KEY" -H "content-type: application/json" \
     -d "{\"image\": \"$(base64 -w0 demo/1_clean.jpg)\"}"
```

Routes: `GET /` page, `POST /analyze`, `POST /decision` (`approve` or `retake`, logged).

## Tests

`scripts/eval_agent.py` doubles as the behavioural test: it asserts nothing by itself
but prints, for real and deliberately damaged plates, which outcome the agent chose and
which tools it called. The live endpoint can be exercised with the four photos in `demo/`.

## Disclosure

The code was written with the help of an AI coding assistant (Claude Code). The deployed
agent contains no language model: its policy is rule-based and deterministic, so the
decisions in a trace can be read directly against the thresholds in `labvision/agent.py`.
