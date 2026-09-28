"""Threaded webcam capture with a short history ring (for click look-back)."""

from __future__ import annotations

import collections
import threading
import time

import cv2
import numpy as np

from .tracker import intrinsics


class Camera:
    def __init__(self, cfg: dict, history_s: float = 1.0):
        c = cfg["camera"]
        # A string index is a video file, looped at its native rate (testing/replay).
        self.is_file = isinstance(c["index"], str)
        if self.is_file:
            self.cap = cv2.VideoCapture(c["index"])
        else:
            self.cap = cv2.VideoCapture(c["index"], cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera index {c['index']}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c["fourcc"]))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
        self.cap.set(cv2.CAP_PROP_FPS, c["fps"])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera opened but returned no frame")
        self.height, self.width = frame.shape[:2]
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.K = intrinsics(self.width, self.height, c["hfov_deg"])
        self._cond = threading.Condition()
        self._latest: tuple[np.ndarray, float, int] | None = None
        self._seq = 0
        self.history = collections.deque(maxlen=max(2, int(history_s * max(self.fps, 30))))
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera")
        self._thread.start()

    def _loop(self):
        period = 1.0 / (self.fps or 30) if self.is_file else 0.0
        while not self._stop:
            ok, frame = self.cap.read()
            t = time.monotonic()
            if not ok:
                if self.is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.01)
                continue
            if period:
                time.sleep(period)
            with self._cond:
                self._seq += 1
                self._latest = (frame, t, self._seq)
                self.history.append(self._latest)
                self._cond.notify_all()

    def next(self, after_seq: int = 0, timeout: float = 1.0):
        """Block until a frame newer than after_seq arrives -> (frame, t, seq)."""
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._latest is not None and self._latest[2] > after_seq, timeout
            )
            if not ok:
                raise TimeoutError("no camera frame")
            return self._latest

    def at(self, t: float):
        """Frame from history closest to (and not after) time t, or None."""
        with self._cond:
            best = None
            for item in self.history:
                if item[1] <= t:
                    best = item
            return best

    def close(self):
        self._stop = True
        self._thread.join(timeout=1)
        self.cap.release()
