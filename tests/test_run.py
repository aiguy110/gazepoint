"""Live-mode state machine with the camera, tracker, pointer, and input stubbed out."""

import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gazepoint import calib, config
from gazepoint.gaze import face_frame, vec_to_angles
from gazepoint.io import OneEuro, PointerDecoder
from gazepoint.monitors import Monitor
from gazepoint.run import GazeSmoother, GazeWorker, Live
from gazepoint.tracker import Observation, Pending

LAPTOP = Monitor("L", "laptop", "eDP-1", 0, 0, 1920, 1200, 300.0, 190.0, True)


class FakePointer:
    def __init__(self, pos=None):
        self.moves = []
        self.rel = []
        self.pos = pos  # None = unreadable, as on Wayland

    def move(self, x, y):
        self.moves.append((x, y))

    def move_rel(self, dx, dy):
        self.rel.append((dx, dy))

    def position(self):
        return self.pos


class FakeTracker:
    """Emits observations of a user at 500 mm looking at a fixed laptop-panel point."""

    def __init__(self, screen, uv):
        self.o = np.array([0.0, -30.0, 500.0])
        self.P = screen.point_cam(*uv)
        self.head_yaw = 0.0

        self.gaze_calls = 0

    def head(self, frame, t, seq, keep_landmarks=False):
        obs = Observation(t, seq, np.eye(3).tolist(), self.o.tolist(), self.head_yaw, 0.0,
                          self.o.tolist(), [0, 0, 1, 1], None, None)
        return Pending(obs, None, None, self.o)

    def gaze(self, p):
        self.gaze_calls += 1
        fR = face_frame(self.o)
        yaw, pitch = np.degrees(vec_to_angles(fR.T @ (self.P - self.o)))
        return replace(p.obs, gaze_raw=[float(yaw), float(pitch)], frame_R=fR.tolist())


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setenv("GAZEPOINT_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))  # no user config
    cfg = config.load()
    cfg["run"].update(log_usage=False, aim_mode="jump", latch_ms=120, gaze_thread=False)
    lv = object.__new__(Live)
    lv.cfg, lv.mons = cfg, [LAPTOP]
    lv.A = calib.A_IDENTITY
    screens = calib.screens_from_monitors([LAPTOP], cfg)
    lv.screens = list(screens.values())
    lv.tracker = FakeTracker(screens["L"], (150.0, 95.0))  # panel centre
    lv.pointer = FakePointer()
    lv.worker = None
    lv.f_yaw, lv.f_pitch = OneEuro(), OneEuro()
    lv.cam = SimpleNamespace(width=64, height=48, K=np.eye(3))
    lv.lock = threading.Lock()
    lv._init_state()
    return lv


def test_press_jump_refine_release(live):
    t = 0.0
    live.step(None, t, 1)
    assert live.pointer.moves == []  # idle: no movement
    live.on_key(True, t)
    for i in range(2, 8):  # 6 frames at 30 fps > latch_ms
        t += 1 / 30
        live.step(None, t, i)
    assert live.state == "held"
    x, y = live.pointer.moves[0]
    assert abs(x - 960) < 5 and abs(y - 600) < 5  # jumped to the panel centre
    live.tracker.head_yaw = 5.0  # turn head to the user's right
    for i in range(8, 40):
        t += 1 / 30
        live.step(None, t, i)
    assert live.pointer.moves[-1][0] > x + 100  # cursor moved right
    live.on_key(False, t)
    assert live.state == "idle"


def test_no_face_explains_once(live, capsys):
    live.tracker.head = lambda *a, **k: None
    live.on_key(True, 0.0)
    for i in range(1, 20):
        live.step(None, i / 30, i)
    live.on_key(False, 1.0)
    out = capsys.readouterr().out
    assert out.count("still waiting to jump: face not detected") == 1
    assert "before a gaze fix" in out
    assert live.pointer.moves == []


def test_reaim_jumps_while_held(live):
    t = 0.0
    live.on_key(True, t)
    for i in range(1, 8):
        t += 1 / 30
        live.step(None, t, i)
    assert live.state == "held"
    x0, _ = live.pointer.moves[-1]
    live.tracker.P = live.screens[0].point_cam(250.0, 95.0)  # look further right
    for i in range(8, 15):  # > latch_ms of looking there before pressing
        t += 1 / 30
        live.step(None, t, i)
    n = len(live.pointer.moves)
    live.on_reaim(True, t)
    assert live.state == "held"  # the gaze from before the press: jumps at once
    x1, y1 = live.pointer.moves[n]
    assert x1 > x0 + 500 and abs(y1 - 600) < 5  # jumped to the new gaze point
    live.on_reaim(False, t)
    assert live.state == "held"  # releasing the re-aim key changes nothing
    live.on_key(False, t)
    assert live.state == "idle"


def test_slow_key_lowers_gain_without_jumping(live):
    t = 0.0
    live.on_key(True, t)
    for i in range(1, 8):
        t += 1 / 30
        live.step(None, t, i)
    live.tracker.head_yaw = 5.0
    for i in range(8, 30):
        t += 1 / 30
        live.step(None, t, i)
    x0 = live.pointer.moves[-1][0]
    live.on_slow(True, t)
    live.tracker.head_yaw = 10.0
    for i in range(30, 60):
        t += 1 / 30
        live.step(None, t, i)
    x1 = live.pointer.moves[-1][0]
    live.on_slow(False, t)
    assert 0 < x1 - x0 < 0.6 * 5 * live.cfg["run"]["gain_px_per_deg"]  # 5 deg at the low gain
    live.on_key(False, t)


def test_reaim_ignored_when_idle(live):
    live.on_reaim(True, 0.0)
    assert live.state == "idle"


FRAME = np.zeros((48, 64, 3), np.uint8)


def test_usage_log_keeps_raw_events_and_context_frames(live):
    live.cfg["run"].update(log_usage=True, context_before_s=0.2, context_after_s=0.1, save_hold_frames=True)
    live._init_state()
    seq, t = 0, 0.0

    def frames(n):
        nonlocal seq, t
        for _ in range(n):
            seq, t = seq + 1, t + 1 / 30
            live.step(FRAME, t, seq)

    frames(30)  # idle: logged, but no images
    live.on_key(True, t)
    frames(10)  # latch, jump, hold
    live.on_pointer({"type": "motion", "t": t, "dev": "touchpad", "dx": 3, "dy": 0})
    live.on_pointer({"type": "button", "t": t, "dev": "kbd mouse", "button": "left", "down": True})
    frames(1)
    live.on_key(False, t)
    frames(30)  # idle again
    live.writer.close()

    ev = [json.loads(ln) for ln in open(live.writer.dir / "events.jsonl")]
    by = lambda k: [e for e in ev if e["type"] == k]
    assert len(by("frame")) == 71 and by("session")
    assert [(e["key"], e["down"]) for e in by("key")] == [("aim", True), ("aim", False)]
    assert len(by("jump")) == 1
    (btn,) = by("button")
    assert btn["state"] == "held" and btn["cursor"] and btn["cursor_trusted"] is False  # touchpad moved it
    saved = sorted(int(p.stem) for p in (live.writer.dir / "frames").iterdir())
    # 0.2 s before the key press, the whole hold, then 0.1 s after the release.
    assert saved == list(range(saved[0], saved[-1] + 1)) and saved[0] in (24, 25) and saved[-1] in (43, 44)


def test_pointer_decoder():
    from evdev import ecodes as e

    out = []
    d = PointerDecoder(out.append, motion_s=0.05)
    for i in range(5):  # 5 motion reports in 40 ms: coalesced into one record
        d.feed("m", e.EV_REL, e.REL_X, 2, i * 0.01)
        d.feed("m", e.EV_SYN, 0, 0, i * 0.01)
    d.feed("m", e.EV_REL, e.REL_Y, -1, 0.06)
    d.feed("m", e.EV_SYN, 0, 0, 0.06)
    d.feed("m", e.EV_KEY, e.BTN_RIGHT, 1, 0.07)
    d.feed("m", e.EV_REL, e.REL_WHEEL, -1, 0.08)
    d.feed("m", e.EV_KEY, e.KEY_A, 1, 0.09)  # keyboard keys are never logged
    assert out == [
        {"type": "motion", "t": 0.06, "dev": "m", "dx": 10, "dy": -1},
        {"type": "button", "t": 0.07, "dev": "m", "button": "right", "down": True},
        {"type": "wheel", "t": 0.08, "dev": "m", "dy": -1},
    ]


def anchor_mode(live, pos=None):
    live.cfg["run"]["aim_mode"] = "anchor"
    live.pointer = FakePointer(pos)


def test_anchor_mode_moves_from_cursor_without_jumping(live):
    anchor_mode(live, pos=(100, 200))
    t = 0.0
    for i in range(1, 5):
        t += 1 / 30
        live.step(None, t, i)
    assert live.tracker.gaze_calls == 0  # gaze net idle until the key is held
    live.on_key(True, t)
    assert live.state == "held" and live.pointer.moves == []
    live.tracker.head_yaw = 5.0
    for i in range(5, 40):
        t += 1 / 30
        live.step(None, t, i)
    assert live.tracker.gaze_calls > 0
    xs = [x for x, _ in live.pointer.moves]
    assert all(abs(x - 100) < 3 * 5 * live.cfg["run"]["gain_px_per_deg"] for x in xs)  # no jump to the panel centre
    assert xs[-1] > 100 + 100 and abs(live.pointer.moves[-1][1] - 200) < 5
    n = len(live.pointer.moves)
    live.on_reaim(True, t)  # jump to the gaze, from frames already seen
    x, y = live.pointer.moves[n]
    assert abs(x - 960) < 5 and abs(y - 600) < 5
    live.on_key(False, t)
    assert live.state == "idle"


def test_anchor_mode_moves_relatively_when_position_unknown(live):
    anchor_mode(live)
    t = 0.0
    live.step(None, t, 1)
    live.on_key(True, t)
    live.tracker.head_yaw = 5.0
    for i in range(2, 40):
        t += 1 / 30
        live.step(None, t, i)
    assert live.pointer.moves == []
    dx = sum(d for d, _ in live.pointer.rel)
    assert dx > 0.8 * 5 * live.cfg["run"]["gain_px_per_deg"]
    live.on_reaim(True, t)
    assert live.pointer.moves and live.anchor is not None  # absolute again after the jump
    live.on_key(False, t)
    # Nothing else moved the pointer since: the next hold starts from the known position.
    live.on_key(True, t)
    assert live.anchor == live.pos and live.anchor is not None
    live.on_key(False, t)
    live.on_pointer({"type": "motion", "t": t, "dev": "touchpad", "dx": 3, "dy": 0})
    live.on_key(True, t)
    assert live.anchor is None  # moved by the touchpad: unknown again
    live.on_key(False, t)


def test_gaze_worker_thread(live):
    live.worker = GazeWorker(live.tracker, live.on_gaze)
    live.worker.start()
    try:
        t = 0.0
        live.on_key(True, t)
        for i in range(1, 30):
            t += 1 / 30
            live.step(None, t, i)
            time.sleep(0.005)
            if live.state == "held":
                break
        assert live.state == "held"
        x, y = live.pointer.moves[0]
        assert abs(x - 960) < 5 and abs(y - 600) < 5
    finally:
        live.worker.stop()
        live.worker.join()


def test_holding_reaim_follows_the_gaze(live):
    anchor_mode(live, pos=(100, 200))
    t = 0.0
    live.on_key(True, t)
    for i in range(1, 10):
        t += 1 / 30
        live.step(None, t, i)
    live.on_reaim(True, t)  # tap: jump to the panel centre
    x, y = live.pointer.moves[-1]
    assert live.state == "held" and abs(x - 960) < 5
    for i in range(10, 20):  # keep holding: past track_after_ms
        t += 1 / 30
        live.step(None, t, i)
    assert live.state == "tracking"
    live.tracker.P = live.screens[0].point_cam(250.0, 95.0)  # look right
    live.tracker.head_yaw = 20.0  # head motion is ignored while tracking
    xs = []
    for i in range(20, 80):
        t += 1 / 30
        live.step(None, t, i)
        xs.append(live.pointer.moves[-1][0])
    assert all(b >= a - 1e-6 for a, b in zip(xs, xs[1:]))  # glides, no overshoot
    target = 250.0 / 300.0 * 1920
    assert abs(xs[-1] - target) < 20 and abs(live.pointer.moves[-1][1] - 600) < 5
    live.on_reaim(False, t)
    assert live.state == "held" and live.anchor == live.pos
    n = len(live.pointer.moves)
    live.tracker.head_yaw = 22.0  # head refines from where tracking stopped
    for i in range(80, 110):
        t += 1 / 30
        live.step(None, t, i)
    assert live.pointer.moves[-1][0] > live.pointer.moves[n - 1][0] + 20
    live.on_key(False, t)
    assert live.state == "idle"


def test_short_reaim_tap_does_not_track(live):
    t = 0.0
    live.on_key(True, t)
    for i in range(1, 8):
        t += 1 / 30
        live.step(None, t, i)
    live.on_reaim(True, t)
    t += 0.1
    live.step(None, t, 8)
    live.on_reaim(False, t)
    for i in range(9, 30):
        t += 1 / 30
        live.step(None, t, i)
    assert live.state == "held"


@pytest.mark.parametrize("kind", ["one_euro", "ema", "median", "none"])
def test_gaze_smoother(kind):
    rc = dict(config.DEFAULTS["run"], track_filter=kind)
    rng = np.random.default_rng(0)
    sm = GazeSmoother(rc, 0.0, 0.0, 0.0)
    out = [sm(1000 + rng.normal(0, 100), 500 + rng.normal(0, 100), i / 30) for i in range(1, 91)]
    end = np.array(out[-30:])
    assert np.all(np.abs(end.mean(0) - [1000, 500]) < 60)  # converged on the new fixation
    if kind != "none":
        assert np.all(end.std(0) < 50)  # and smoothed (raw noise is 100 px)
