"""Build and publish the two Lambda layers (OpenCV 5 does not fit one 50 MB upload).

  lv-cv2         : cv2/ (the OpenCV 5 engine)
  lv-libs_numpy  : opencv_python_headless.libs/ + numpy

Linux x86_64 / CPython 3.13 wheels are downloaded from PyPI at the versions pinned
in requirements.lock.txt. Uses the AWS profile 'labvision'.
Run from the project root:  python deploy/build_layers.py
"""
import glob
import os
import subprocess
import sys
import tempfile
import zipfile

import boto3

lam = boto3.Session(profile_name="labvision", region_name="us-east-1").client("lambda")
tmp = tempfile.mkdtemp()
subprocess.check_call([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
                       "--platform", "manylinux2014_x86_64", "--platform", "manylinux_2_28_x86_64",
                       "--python-version", "3.13", "--implementation", "cp", "-d", tmp,
                       "opencv-python-headless==5.0.0.93", "numpy==2.5.4"])
cv = [w for w in glob.glob(tmp + "/*.whl") if "opencv" in w][0]
npw = [w for w in glob.glob(tmp + "/*.whl") if "numpy" in w][0]


def make(name, parts):
    out = os.path.join(tmp, name + ".zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for wheel, keep in parts:
            w = zipfile.ZipFile(wheel)
            for i in w.infolist():
                if not i.is_dir() and keep(i.filename):
                    z.writestr("python/" + i.filename, w.read(i.filename))
    assert os.path.getsize(out) < 50_000_000, "layer over the 50 MB direct-upload limit"
    arn = lam.publish_layer_version(LayerName=name, Content={"ZipFile": open(out, "rb").read()},
                                    CompatibleRuntimes=["python3.13"],
                                    CompatibleArchitectures=["x86_64"])["LayerVersionArn"]
    print(arn)
    return arn


a = make("lv-cv2", [(cv, lambda f: f.startswith("cv2/"))])
b = make("lv-libs_numpy", [(cv, lambda f: f.startswith("opencv_python_headless.libs/")),
                           (npw, lambda f: True)])
os.makedirs("out", exist_ok=True)
open("out/layers.txt", "w").write(a + "\n" + b + "\n")
