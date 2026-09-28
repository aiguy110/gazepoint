"""Build an ONNX ResNet-18 from a timm/torchvision state dict, without torch.

Used for the ETH-XGaze gaze model, which is published only as PyTorch weights
(hysts/ptgaze-eth-xgaze-resnet18 on Hugging Face). The laptop has no torch, so
`download-models` converts it here. BatchNorm is folded into the preceding conv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from safetensors.numpy import load_file


def build(sd: dict[str, np.ndarray], out_dim: int) -> onnx.ModelProto:
    nodes, inits = [], []

    def const(name, arr):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=np.float32), name))
        return name

    def conv_bn(x, conv, bn, stride, pad, relu):
        w = sd[f"{conv}.weight"]
        scale = sd[f"{bn}.weight"] / np.sqrt(sd[f"{bn}.running_var"] + 1e-5)
        W = const(f"{conv}.W", w * scale[:, None, None, None])
        B = const(f"{conv}.B", sd[f"{bn}.bias"] - sd[f"{bn}.running_mean"] * scale)
        y = f"{conv}.out"
        k = w.shape[2]
        nodes.append(helper.make_node(
            "Conv", [x, W, B], [y], kernel_shape=[k, k], strides=[stride, stride], pads=[pad] * 4))
        if relu:
            nodes.append(helper.make_node("Relu", [y], [y + ".relu"]))
            y += ".relu"
        return y

    x = conv_bn("input", "conv1", "bn1", 2, 3, True)
    nodes.append(helper.make_node("MaxPool", [x], ["pool"], kernel_shape=[3, 3], strides=[2, 2], pads=[1] * 4))
    x = "pool"
    for li in range(1, 5):
        for bi in range(2):
            p = f"layer{li}.{bi}"
            stride = 2 if li > 1 and bi == 0 else 1
            h = conv_bn(x, f"{p}.conv1", f"{p}.bn1", stride, 1, True)
            h = conv_bn(h, f"{p}.conv2", f"{p}.bn2", 1, 1, False)
            skip = x
            if f"{p}.downsample.0.weight" in sd:
                skip = conv_bn(x, f"{p}.downsample.0", f"{p}.downsample.1", stride, 0, False)
            nodes.append(helper.make_node("Add", [h, skip], [f"{p}.sum"]))
            nodes.append(helper.make_node("Relu", [f"{p}.sum"], [f"{p}.out"]))
            x = f"{p}.out"
    nodes.append(helper.make_node("GlobalAveragePool", [x], ["gap"]))
    nodes.append(helper.make_node("Flatten", ["gap"], ["feat"], axis=1))
    nodes.append(helper.make_node(
        "Gemm", ["feat", const("fc.W", sd["fc.weight"]), const("fc.B", sd["fc.bias"])], ["output"], transB=1))
    graph = helper.make_graph(
        nodes, "resnet18", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3, 224, 224])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, out_dim])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def convert(safetensors_path: Path, onnx_path: Path) -> None:
    sd = load_file(str(safetensors_path))
    onnx.save(build(sd, sd["fc.weight"].shape[0]), str(onnx_path))
