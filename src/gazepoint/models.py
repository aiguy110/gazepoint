"""Model / asset download. Everything lands in GAZEPOINT_HOME/models."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from . import config

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
CANONICAL_FACE_URL = (
    "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/"
    "mediapipe/modules/face_geometry/data/canonical_face_model.obj"
)
# yakhyo/gaze-estimation (MIT): L2CS-style nets trained on Gaze360, 90 bins x 4 deg.
GAZE_URL = "https://github.com/yakhyo/gaze-estimation/releases/download/weights/{name}_gaze.onnx"
GAZE360_MODELS = ("mobileone_s0", "mobilenetv2", "resnet18", "resnet34", "resnet50")
# hysts/pytorch_mpiigaze_demo (MIT code; ETH-XGaze data is CC BY-NC-SA): ResNet-18 on
# normalised ETH-XGaze faces. Much more robust to lighting than the Gaze360 nets.
# Published as PyTorch weights only; converted to ONNX at download time.
ETH_URL = "https://huggingface.co/hysts/ptgaze-eth-xgaze-resnet18/resolve/main/model.safetensors"
ETH_MODELS = ("ethxgaze_resnet18",)
GAZE_MODELS = ETH_MODELS + GAZE360_MODELS


def is_eth(name: str) -> bool:
    return name in ETH_MODELS


def face_landmarker_path() -> Path:
    return config.models_dir() / "face_landmarker.task"


def canonical_face_path() -> Path:
    return config.models_dir() / "canonical_face_model.obj"


def gaze_model_path(name: str) -> Path:
    return config.models_dir() / f"{name}_gaze.onnx"


def _fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}")
        return
    print(f"  fetching {dest.name} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def download(gaze_models: list[str] | None = None) -> None:
    cfg = config.load()
    names = gaze_models or [cfg["gaze"]["model"]]
    for n in names:
        if n not in GAZE_MODELS:
            sys.exit(f"unknown gaze model {n!r}; choose from {GAZE_MODELS}")
    print(f"models dir: {config.models_dir()}")
    _fetch(FACE_LANDMARKER_URL, face_landmarker_path())
    _fetch(CANONICAL_FACE_URL, canonical_face_path())
    for n in names:
        if is_eth(n):
            dest = gaze_model_path(n)
            if dest.exists():
                print(f"  have {dest.name}")
                continue
            weights = config.models_dir() / f"{n}.safetensors"
            _fetch(ETH_URL, weights)
            from .resnet_onnx import convert

            print(f"  converting to {dest.name}")
            convert(weights, dest)
        else:
            _fetch(GAZE_URL.format(name=n), gaze_model_path(n))


def ensure(cfg: dict) -> None:
    """Download whatever `cfg` needs that's missing (e.g. after a `git pull` without bootstrap)."""
    if check(cfg):
        print("missing models; downloading", file=sys.stderr)
        download([cfg["gaze"]["model"]])


def check(cfg: dict) -> list[str]:
    """Return a list of missing asset paths."""
    need = [face_landmarker_path(), canonical_face_path(), gaze_model_path(cfg["gaze"]["model"])]
    return [str(p) for p in need if not p.exists()]
