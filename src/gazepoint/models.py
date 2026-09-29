"""Model / asset download. Everything lands in GAZEPOINT_HOME/models."""

from __future__ import annotations

import hashlib
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
# UniGaze (Qin et al. 2025; weights CC BY-NC-SA, non-commercial): ViT-B/16 MAE-pretrained on
# large face datasets, then trained on normalised ETH-XGaze etc. About 2x more accurate than
# ETH-XGaze ResNet-18 on recorded sessions, but ~8x the compute. Exported from the
# `unigaze` package to ONNX on the dev box (input/output as for ETH-XGaze) and self-hosted.
UNIGAZE_URL = "https://outpost.kleinflask.us/gazepoint/{name}.onnx"
UNIGAZE_SHA256 = {
    "unigaze_b16_joint": "429870c7b1433520e43719eac89eae97a40c64898f1ec2f8bfe5991b14f30951",
}
# Nets taking ETH-XGaze-normalised faces (gaze.normalize_face).
NORMALIZED_MODELS = ("ethxgaze_resnet18",) + tuple(UNIGAZE_SHA256)
GAZE_MODELS = NORMALIZED_MODELS + GAZE360_MODELS


def is_normalized(name: str) -> bool:
    return name in NORMALIZED_MODELS


def face_landmarker_path() -> Path:
    return config.models_dir() / "face_landmarker.task"


def canonical_face_path() -> Path:
    return config.models_dir() / "canonical_face_model.obj"


def gaze_model_path(name: str) -> Path:
    return config.models_dir() / f"{name}_gaze.onnx"


def _fetch(url: str, dest: Path, sha256: str | None = None) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}")
        return
    print(f"  fetching {dest.name} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = [0]

    def progress(blocks, bs, total):
        mb = blocks * bs // 20_000_000  # every 20 MB
        if total > 50_000_000 and mb > last[0]:
            last[0] = mb
            print(f"    {blocks * bs / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)

    urllib.request.urlretrieve(url, tmp, progress)
    if sha256:
        h = hashlib.sha256()
        with open(tmp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != sha256:
            tmp.unlink()
            sys.exit(f"{dest.name}: checksum mismatch (got {h.hexdigest()}); try again")
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
        fetch_gaze(n)


def fetch_gaze(n: str) -> None:
    dest = gaze_model_path(n)
    if n in UNIGAZE_SHA256:
        _fetch(UNIGAZE_URL.format(name=n), dest, UNIGAZE_SHA256[n])
    elif n == "ethxgaze_resnet18":
        if dest.exists():
            print(f"  have {dest.name}")
            return
        weights = config.models_dir() / f"{n}.safetensors"
        _fetch(ETH_URL, weights)
        from .resnet_onnx import convert

        print(f"  converting to {dest.name}")
        convert(weights, dest)
    else:
        _fetch(GAZE_URL.format(name=n), dest)


def ensure(cfg: dict) -> None:
    """Download whatever `cfg` needs that's missing (e.g. after a `git pull` without bootstrap)."""
    if check(cfg):
        print("missing models; downloading", file=sys.stderr)
        download([cfg["gaze"]["model"]])


def check(cfg: dict) -> list[str]:
    """Return a list of missing asset paths."""
    need = [face_landmarker_path(), canonical_face_path(), gaze_model_path(cfg["gaze"]["model"])]
    return [str(p) for p in need if not p.exists()]
