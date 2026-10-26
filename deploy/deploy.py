"""Deploy the agent: SNS topic, Lambda function (+ two layers), Function URL.

Idempotent: re-running updates code and config in place. Prints the endpoint; the
API token is generated once and kept in out/deploy.json (git-ignored), never
printed. Run from the project root:  python deploy/deploy.py
"""
import io
import json
import os
import secrets
import zipfile

import boto3

sess = boto3.Session(profile_name="labvision", region_name="us-east-1")
lam, sns = sess.client("lambda"), sess.client("sns")
acct = sess.client("sts").get_caller_identity()["Account"]
NAME = "lv-agent"
ROLE = f"arn:aws:iam::{acct}:role/lab-vision-lambda-role"
if os.path.exists("out/layers.txt"):
    LAYERS = open("out/layers.txt").read().split()
else:
    LAYERS = [f"arn:aws:lambda:us-east-1:{acct}:layer:lv-cv2:1",
              f"arn:aws:lambda:us-east-1:{acct}:layer:lv-libs_numpy:1"]
os.makedirs("out", exist_ok=True)
state_path = "out/deploy.json"
state = json.load(open(state_path)) if os.path.exists(state_path) else {}
state.setdefault("token", secrets.token_urlsafe(24))
topic = sns.create_topic(Name="lab-vision-alerts")["TopicArn"]

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for f in ("lambda_function.py", "page.py"):
        z.write(f)
    for f in os.listdir("labvision"):
        if f.endswith((".py", ".json")):
            z.write(os.path.join("labvision", f))

cfg = dict(Runtime="python3.13", Handler="lambda_function.handler", Role=ROLE,
           Timeout=60, MemorySize=512, Layers=LAYERS,
           Environment={"Variables": {"API_TOKEN": state["token"], "SNS_TOPIC_ARN": topic}})
try:
    lam.get_function(FunctionName=NAME)
    exists = True
except lam.exceptions.ResourceNotFoundException:
    exists = False
if exists:
    lam.update_function_configuration(FunctionName=NAME, **cfg)
    lam.get_waiter("function_updated_v2").wait(FunctionName=NAME)
    lam.update_function_code(FunctionName=NAME, ZipFile=buf.getvalue())
else:
    lam.create_function(FunctionName=NAME, Code={"ZipFile": buf.getvalue()},
                        Architectures=["x86_64"], **cfg)
lam.get_waiter("function_active_v2").wait(FunctionName=NAME)
lam.get_waiter("function_updated_v2").wait(FunctionName=NAME)
try:   # cost guard: at most 2 simultaneous runs
    lam.put_function_concurrency(FunctionName=NAME, ReservedConcurrentExecutions=2)
except Exception as e:
    print("concurrency cap not set:", str(e)[:160])
try:
    url = lam.create_function_url_config(FunctionName=NAME, AuthType="NONE")["FunctionUrl"]
except lam.exceptions.ResourceConflictException:
    url = lam.get_function_url_config(FunctionName=NAME)["FunctionUrl"]
# A public Function URL needs BOTH statements (AWS requirement since late 2025).
for sid, extra in (("PublicUrl", dict(Action="lambda:InvokeFunctionUrl", FunctionUrlAuthType="NONE")),
                   ("PublicUrlInvoke", dict(Action="lambda:InvokeFunction", InvokedViaFunctionUrl=True))):
    try:
        lam.add_permission(FunctionName=NAME, StatementId=sid, Principal="*", **extra)
    except lam.exceptions.ResourceConflictException:
        pass
state.update(url=url, topic=topic, function=NAME)
json.dump(state, open(state_path, "w"), indent=1)
print("endpoint:", url)
print("sns topic:", topic)
