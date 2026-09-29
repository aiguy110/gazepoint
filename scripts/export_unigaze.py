"""Export a UniGaze model to ONNX for gazepoint (dev box only; not a gazepoint dependency).

Needs an env with `unigaze` (github.com/ut-vision/UniGaze), torch 2.0.x, timm 0.3.2,
onnx and onnxruntime. Input: ETH-XGaze-normalised 224x224 face, RGB, ImageNet mean/std;
output: (pitch, yaw) radians, as for the ETH-XGaze ResNet-18. Checks the export against
torch on random input, then prints the sha256 for models.UNIGAZE_SHA256. Publish with:

    rsync <name>.onnx outpost:/tmp/ && ssh outpost sudo mv /tmp/<name>.onnx /srv/outpost/gazepoint/

    python export_unigaze.py unigaze_b16_joint
"""
import hashlib
import sys

import numpy as np
import onnxruntime as ort
import torch
import unigaze


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)["pred_gaze"]


name = sys.argv[1]
out = f"{name}.onnx"
w = Wrap(unigaze.load(name)).eval()
x = torch.randn(4, 3, 224, 224)
torch.onnx.export(w, x[:1], out, input_names=["input"], output_names=["output"], opset_version=17,
                  dynamic_axes={"input": {0: "n"}, "output": {0: "n"}})
with torch.no_grad():
    ref = w(x).numpy()
s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
got = s.run(None, {"input": x.numpy()})[0]
print(f"max |onnx - torch|: {np.degrees(np.abs(got - ref)).max():.2e} deg")
print("sha256", hashlib.sha256(open(out, "rb").read()).hexdigest())
