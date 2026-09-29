import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from gazepoint import gaze


@pytest.fixture
def tiny_onnx(tmp_path, monkeypatch):
    """N x 3 x H x W (dynamic) -> per-channel means, N x 3."""
    monkeypatch.setenv("GAZEPOINT_HOME", str(tmp_path))
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 3, "h", "w"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n", 3])
    g = helper.make_graph(
        [helper.make_node("GlobalAveragePool", ["x"], ["p"]), helper.make_node("Flatten", ["p"], ["y"])],
        "tiny", [x], [y],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    path = tmp_path / "tiny.onnx"
    onnx.save(m, str(path))
    return str(path)


def _check(engine):
    x = np.random.default_rng(0).random((1, 3, 224, 224), dtype=np.float32)
    (y,) = engine(x)
    np.testing.assert_allclose(y[0], x.mean(axis=(0, 2, 3)), atol=1e-3)


def test_onnxruntime(tiny_onnx):
    e = gaze.Engine(tiny_onnx, "onnxruntime")
    assert e.device.startswith("onnxruntime") and e.size == (224, 224)
    _check(e)


def test_auto_falls_back_without_npu(tiny_onnx, monkeypatch):
    monkeypatch.setattr(gaze, "openvino_devices", lambda: ["CPU"])
    e = gaze.Engine(tiny_onnx, "auto")
    assert e.device.startswith("onnxruntime")
    _check(e)


def test_explicit_missing_device_fails(tiny_onnx, monkeypatch):
    monkeypatch.setattr(gaze, "openvino_devices", lambda: ["CPU"])
    with pytest.raises(RuntimeError, match="no NPU"):
        gaze.Engine(tiny_onnx, "npu")
    with pytest.raises(ValueError):
        gaze.Engine(tiny_onnx, "tpu")


@pytest.mark.skipif("NPU" not in gaze.openvino_devices(), reason="no usable NPU")
def test_npu(tiny_onnx):
    e = gaze.Engine(tiny_onnx, "npu")
    assert e.device == "OpenVINO NPU"
    _check(e)
