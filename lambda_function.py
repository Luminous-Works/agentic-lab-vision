"""AWS Lambda entry point (Function URL). Routes:
  GET  /          the review page (upload a plate photo)
  POST /analyze   {"image": "<base64 jpeg/png>"}  header x-api-key  -> verdict + trace + overlay
  POST /decision  {"id": "...", "decision": "approve"|"retake"}      -> logged human decision
"""
import base64
import hmac
import json
import os
import time
import uuid

import cv2
import numpy as np

from labvision import agent as _agent
from labvision.agent import review
from labvision.colonies import analyse

TOKEN = os.environ.get("API_TOKEN", "")
TOPIC = os.environ.get("SNS_TOPIC_ARN", "")
MAX_BYTES = 5_000_000
_sns = None


def _notify(msg):
    """Log the alert, and publish it to SNS when a topic is configured."""
    global _sns
    print(json.dumps({"alert": msg}))
    if TOPIC:
        import boto3
        _sns = _sns or boto3.client("sns")
        _sns.publish(TopicArn=TOPIC, Subject="Agentic Lab Vision alert", Message=msg)


def _resp(code, body, ctype="application/json"):
    return {"statusCode": code,
            "headers": {"content-type": ctype, "cache-control": "no-store"},
            "body": body if isinstance(body, str) else json.dumps(body)}


def _ok_token(headers):
    got = (headers or {}).get("x-api-key", "")
    return bool(TOKEN) and hmac.compare_digest(got.encode(), TOKEN.encode())


def handler(event, context):
    http = event.get("requestContext", {}).get("http", {})
    method, path = http.get("method", "GET"), event.get("rawPath", "/")
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8", "replace")

    if method == "GET" and path == "/":
        from page import PAGE
        return _resp(200, PAGE, "text/html; charset=utf-8")

    if method == "POST" and path in ("/analyze", "/decision"):
        if not _ok_token(headers):
            return _resp(401, {"error": "missing or wrong x-api-key"})
        try:
            req = json.loads(body)
        except ValueError:
            return _resp(400, {"error": "body must be JSON"})

        if path == "/decision":
            d = req.get("decision")
            if d not in ("approve", "retake"):
                return _resp(400, {"error": "decision must be approve or retake"})
            print(json.dumps({"human_decision": d, "id": str(req.get("id"))[:64], "t": time.time()}))
            return _resp(200, {"recorded": d, "id": req.get("id")})

        raw = base64.b64decode(req.get("image", ""), validate=False)
        if not raw or len(raw) > MAX_BYTES:
            return _resp(413 if raw else 400, {"error": f"image missing or over {MAX_BYTES} bytes"})
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return _resp(400, {"error": "could not decode image"})
        t0 = time.time()
        res = review(img, notify=_notify)
        out = res.to_dict()
        out["id"] = str(uuid.uuid4())
        out["ms"] = int((time.time() - t0) * 1000)
        out["opencv"] = cv2.__version__
        a = analyse(res.image, draw=True, exclude=res.exclude, **_agent.PARAMS)
        _, jpg = cv2.imencode(".jpg", a.overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out["overlay"] = base64.b64encode(jpg).decode()
        print(json.dumps({k: out[k] for k in ("id", "outcome", "count", "reason", "ms")}))
        return _resp(200, out)

    return _resp(404, {"error": "not found"})
