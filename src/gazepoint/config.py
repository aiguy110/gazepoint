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
        "index": 0,  # V4L2 index, /dev/videoN, device name, or a video file (looped, for testing)
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
        # ethxgaze_resnet18 (ETH-XGaze, normalised faces; far more robust to
        # lighting), unigaze_b16_joint (about twice as accurate again, ~8x the
        # compute: ~60 ms/frame on 8 cores; use with run.gaze_thread), or a
        # Gaze360 net: mobileone_s0 | mobilenetv2 | resnet18 | resnet34 |
        # resnet50. The per-user correction is fit per model.
        "model": "ethxgaze_resnet18",
        # onnxruntime intra-op threads for the gaze net; 0 = its default (all
        # physical cores). `gazepoint bench` times a few settings.
        "threads": 0,
        # Gaze360 nets only: side of the square face crop fed to the net,
        # relative to the landmark bounding box (they expect RetinaFace-ish boxes).
        "crop_scale": 1.0,
        # Gaze360 nets only: interpret model output relative to a virtual camera
        # looking straight at the face (how Gaze360 crops are rectified) rather
        # than the real camera axis.
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
        # Re-show a target (up to fixation_retries times) when its capture looks
        # like the user wasn't fixating: raw gaze too scattered (75th pct distance
        # from the median) or the first and second halves disagree (late saccade,
        # looked away). Net noise alone is ~5 deg at the 75th pct.
        "fixation_spread_deg": 9.0,
        "fixation_drift_deg": 10.0,
        "fixation_min_frames": 5,
        "fixation_retries": 2,
        "background": "#303030",
        "jpeg_quality": 92,
    },
    "calibrate": {
        # Refuse to save a fit (keep the previous calibration) when the gaze
        # correction has collapsed (smallest singular value of A below
        # min_gain: the fit ignores gaze and just places screens around a
        # constant direction) or too many frames land on the wrong screen.
        # This happens when the net's raw gaze doesn't track the targets,
        # e.g. a backlit or shadowed face. `calibrate --force` saves anyway.
        "min_gain": 0.3,
        "max_wrong_screen": 0.3,
    },
    "run": {
        "hotkey": "KEY_RIGHTCTRL",
        # What pressing the hotkey does:
        #   jump:   jump to the gaze (median over the next latch_ms), then head
        #           motion refines.
        #   anchor: head motion moves the cursor from where it is, with no jump;
        #           reaim_key jumps to the gaze. The gaze net only runs while
        #           the hotkey is held.
        "aim_mode": "jump",
        # While the hotkey is held, pressing this jumps to the gaze (again): the
        # median over the latch_ms *before* the press, so the jump is immediate
        # (e.g. to carry a drag to its destination). "" disables it.
        "reaim_key": "",
        "latch_ms": 120,
        # Run the gaze net on its own thread, on the newest frame, so a slow
        # net (UniGaze) doesn't hold back head tracking. Gaze then updates at
        # the net's own rate.
        "gaze_thread": True,
        # Cursor px per degree of head rotation while the key is held.
        "gain_px_per_deg": 35.0,
        # Lower gain for fine positioning while this key is held ("" disables).
        "slow_key": "",
        "gain_slow_px_per_deg": 12.0,
        "pointer_backend": "auto",  # auto | x11 | uinput
        "one_euro_min_cutoff": 1.0,
        "one_euro_beta": 0.05,
        "log_usage": True,
        # With log_usage: events.jsonl gets every frame's observation plus key,
        # mouse button, wheel, touch, and motion events, so labels can be
        # rederived offline (e.g. drop abandoned holds, label by click position).
        # Webcam frames are saved for the whole of each hold and for this long
        # around every key/button event (~100 KB each at 720p).
        "context_before_s": 0.5,
        "context_after_s": 0.3,
        "save_hold_frames": True,
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
