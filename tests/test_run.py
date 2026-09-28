"""Live-mode state machine with the camera, tracker, pointer, and input stubbed out."""

import numpy as np
import pytest

from gazepoint import calib, config
from gazepoint.gaze import face_frame, vec_to_angles
from gazepoint.io import OneEuro
from gazepoint.monitors import Monitor
from gazepoint.run import Live
from gazepoint.tracker import Observation

LAPTOP = Monitor("L", "laptop", "eDP-1", 0, 0, 1920, 1200, 300.0, 190.0, True)


class FakePointer:
    def __init__(self):
        self.moves = []

    def move(self, x, y):
        self.moves.append((x, y))


class FakeTracker:
    """Emits observations of a user at 500 mm looking at a fixed laptop-panel point."""

    def __init__(self, screen, uv):
        self.o = np.array([0.0, -30.0, 500.0])
        self.P = screen.point_cam(*uv)
        self.head_yaw = 0.0

    def process(self, frame, t, seq, keep_landmarks=False):
        fR = face_frame(self.o)
        yaw, pitch = np.degrees(vec_to_angles(fR.T @ (self.P - self.o)))
        return Observation(t, seq, np.eye(3).tolist(), self.o.tolist(), self.head_yaw, 0.0,
                           self.o.tolist(), [0, 0, 1, 1], [float(yaw), float(pitch)], fR.tolist())


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setenv("GAZEPOINT_HOME", str(tmp_path))
    cfg = config.load()
    cfg["run"]["log_usage"] = False
    lv = object.__new__(Live)
    lv.cfg, lv.mons = cfg, [LAPTOP]
    lv.A = calib.A_IDENTITY
    screens = calib.screens_from_monitors([LAPTOP], cfg)
    lv.screens = list(screens.values())
    lv.tracker = FakeTracker(screens["L"], (150.0, 95.0))  # panel centre
    lv.pointer = FakePointer()
    lv.f_yaw, lv.f_pitch = OneEuro(), OneEuro()
    import threading

    lv.lock = threading.Lock()
    lv.state, lv.latch, lv.writer, lv.events = "idle", [], None, 0
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
