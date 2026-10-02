"""Configuration and on-disk locations.

Everything stateful lives under GAZEPOINT_HOME (default ~/.local/share/gazepoint):

    models/                 downloaded weights (see models.py)
    sessions/<id>/          recorded data: session.json, samples.jsonl, frames/*.jpg
    profiles/<hash>.json    monitor constellations, keyed by the set of external monitors
    user.json               per-user gaze correction (independent of desk setup)
    users/<name>/user.json  the same for user profile <name> (`--profile`)
    current.json            camera pose for the current desk session

Monitor constellations and the camera pose are physical, so every user profile
shares them; only the gaze correction is per user.

Pointing GAZEPOINT_HOME at a pulled data dir (data/<host>) lets you re-run
calibrate/eval on the dev box against data gathered on the laptop.

Defaults, with comments on every setting, are in config.toml next to this file;
user overrides live in ~/.config/gazepoint/config.toml (any subset of it).
"""

from __future__ import annotations

import copy
import os
import re
import tomllib
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name("config.toml")  # commented; also the example user config
with open(DEFAULT_PATH, "rb") as _f:
    DEFAULTS: dict = tomllib.load(_f)


def home() -> Path:
    p = Path(os.environ.get("GAZEPOINT_HOME", Path.home() / ".local/share/gazepoint"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_profile() -> str | None:
    """The user profile (`--profile`, or GAZEPOINT_PROFILE); None = the default user."""
    return os.environ.get("GAZEPOINT_PROFILE") or None


def set_user_profile(name: str):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"bad profile name {name!r}: use letters, digits, '_', '.', '-'")
    os.environ["GAZEPOINT_PROFILE"] = name


def profile_flag() -> str:
    """' --profile <name>' for commands suggested in messages, or ''."""
    return f" --profile {user_profile()}" if user_profile() else ""


def user_dir() -> Path:
    """Where the current user profile's gaze correction lives."""
    name = user_profile()
    if not name:
        return home()
    d = home() / "users" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


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


def _merge(base: dict, over: dict, where: str = "") -> dict:
    for k, v in over.items():
        if k not in base:
            print(f"warning: {config_path()}: unknown setting {where}{k} (ignored by gazepoint)")
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v, f"{where}{k}.")
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
