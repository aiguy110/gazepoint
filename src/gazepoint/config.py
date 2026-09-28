"""Configuration and on-disk locations.

Everything stateful lives under GAZEPOINT_HOME (default ~/.local/share/gazepoint):

    models/                 downloaded weights (see models.py)
    sessions/<id>/          recorded data: session.json, samples.jsonl, frames/*.jpg
    profiles/<hash>.json    monitor constellations, keyed by the set of external monitors
    user.json               per-user gaze correction (independent of desk setup)
    current.json            camera pose for the current desk session

Pointing GAZEPOINT_HOME at a pulled data dir (data/<host>) lets you re-run
calibrate/eval on the dev box against data gathered on the laptop.

User overrides live in ~/.config/gazepoint/config.toml (any subset of DEFAULTS).
"""

from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path

DEFAULTS: dict = {
    "camera": {
        "index": 0,  # V4L2 index, or a video file path (looped) for testing
        "width": 1280,
        "height": 720,
        "fps": 30,
        "fourcc": "MJPG",
        # Horizontal field of view used to build intrinsics. Laptop webcams are
        # typically 65-75 deg horizontal. Errors here mostly scale depth, which
        # calibration partially absorbs.
        "hfov_deg": 70.0,
    },
    "gaze": {
        "model": "mobileone_s0",  # mobileone_s0 | mobilenetv2 | resnet18 | resnet34 | resnet50
        # Side of the square face crop fed to the gaze net, relative to the
        # landmark bounding box. Pretrained nets expect RetinaFace-ish boxes.
        "crop_scale": 1.0,
        # Interpret model output relative to a virtual camera looking straight
        # at the face (how Gaze360 crops are rectified) rather than the real
        # camera axis.
        "face_centered": True,
    },
    "laptop": {
        # Camera position relative to the top-center of the built-in panel's
        # active area, in mm (+x toward the user's right, +y up). The lid holds
        # both, so this never changes.
        "camera_above_panel_mm": 8.0,
        "camera_right_of_center_mm": 0.0,
    },
    "collect": {
        "points_full": 9,
        "points_quick": 5,
        "settle_ms": 800,
        "capture_ms": 700,
        "background": "#303030",
        "jpeg_quality": 92,
    },
    "run": {
        "hotkey": "KEY_RIGHTCTRL",
        "latch_ms": 120,
        # Cursor px per degree of head rotation while the key is held.
        "gain_px_per_deg": 35.0,
        "pointer_backend": "auto",  # auto | x11 | uinput
        "one_euro_min_cutoff": 1.0,
        "one_euro_beta": 0.05,
        "log_usage": True,
        # X11 only: log (frame ~150ms before click, cursor position) pairs.
        "log_clicks": False,
        "click_lookback_ms": 150,
    },
}


def home() -> Path:
    p = Path(os.environ.get("GAZEPOINT_HOME", Path.home() / ".local/share/gazepoint"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    d = home() / "models"
    d.mkdir(exist_ok=True)
    return d


def sessions_dir() -> Path:
    d = home() / "sessions"
    d.mkdir(exist_ok=True)
    return d


def profiles_dir() -> Path:
    d = home() / "profiles"
    d.mkdir(exist_ok=True)
    return d


def config_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "gazepoint" / "config.toml"


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load() -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    p = config_path()
    if p.exists():
        with open(p, "rb") as f:
            _merge(cfg, tomllib.load(f))
    return cfg
