"""Live mode: hold the hotkey -> cursor jumps to gaze, head motion refines.

State machine (driven by camera frames):

  idle --key down--> latching: gather gaze hits for latch_ms
  latching --------> held: warp to median hit (the anchor), remember head pose
  held: cursor = anchor + gain * (head delta)   (head angles One-Euro filtered)
  held --key up----> idle: log (latch frames, anchor, final cursor) as a label
  held/latching --reaim key down--> latching: log the segment so far, latch again

The final cursor position is usually where the user wanted to be, so each use
produces a labelled sample (samples.jsonl) for later recalibration/training.
It isn't always: holds get abandoned or retried. So usage sessions also keep a
raw stream in events.jsonl, from which labels can be rederived offline:

  session  t, wall            clock alignment (wall = time.time() at monotonic t)
  frame    t, seq, state, obs, gaze [x, y] (current calibration), cursor
  key      t, key (aim|reaim|slow), down, state
  jump     t, anchor, n (gaze frames in the median)
  button   t, dev, button, down, cursor, cursor_trusted, state
  wheel    t, dev, dx|dy          touch  t, dev, down
  motion   t, dev, dx, dy (relative motion from another pointing device)

`cursor` is where gazepoint last put the pointer; `cursor_trusted` is false
once another device has moved it since (on Wayland the real position is
unknowable). On X11 button records also carry `cursor_actual`. Webcam frames
are saved as frames/<seq:08d>.jpg during holds and around key/button events
(some only after the fact, from the ring buffer), so check for the file.
"""

from __future__ import annotations

import collections
import threading
import time

import numpy as np

from . import calib, record
from .calib import gaze_dir
from .camera import Camera
from .geometry import Screen, hit_screens
from .io import InputListener, OneEuro, make_pointer
from .monitors import Monitor, detect, profile_key, session_type
from .tracker import Tracker


def gaze_px(A, screens: list[Screen], obs) -> tuple[Screen, float, float] | None:
    d = gaze_dir(A, np.array(obs.gaze_raw), np.array(obs.frame_R))[0]
    hit = hit_screens(screens, np.array(obs.eye_origin), d)
    if hit is None:
        return None
    s, u, v, _ = hit
    u = min(max(u, 0.0), s.size_mm[0])
    v = min(max(v, 0.0), s.size_mm[1])
    return s, *s.mm_to_px(u, v)


def clamp_desktop(mons: list[Monitor], x: float, y: float) -> tuple[float, float, Monitor]:
    best = None
    for m in mons:
        cx = min(max(x, m.x), m.x + m.w - 1)
        cy = min(max(y, m.y), m.y + m.h - 1)
        d = (cx - x) ** 2 + (cy - y) ** 2
        if best is None or d < best[0]:
            best = (d, cx, cy, m)
    return best[1], best[2], best[3]


class Live:
    def __init__(self, cfg: dict, log_clicks: bool | None = None):
        self.cfg = cfg
        rc = cfg["run"]
        self.mons = detect()
        self.A, self.screens = calib.load_runtime(self.mons, cfg)
        self.cam = Camera(cfg)
        self.tracker = Tracker(cfg, self.cam.K)
        self.pointer = make_pointer(rc["pointer_backend"], self.mons)
        self.f_yaw = OneEuro(rc["one_euro_min_cutoff"], rc["one_euro_beta"])
        self.f_pitch = OneEuro(rc["one_euro_min_cutoff"], rc["one_euro_beta"])
        self.lock = threading.Lock()
        self._init_state()
        log_clicks = rc["log_clicks"] if log_clicks is None else log_clicks
        if log_clicks and session_type() != "x11":
            print("click logging needs X11 (cursor position is not observable on Wayland); disabled")
            log_clicks = False
        self.listener = InputListener(rc["hotkey"], self.on_key, self.on_click if log_clicks else None,
                                      rc["reaim_key"], self.on_reaim,
                                      self.on_pointer if rc["log_usage"] else None,
                                      rc["slow_key"], self.on_slow)

    def _init_state(self):
        rc = self.cfg["run"]
        self.state = "idle"
        self.latch_t = 0.0
        self.latch: list = []  # (frame, obs, hit)
        self.latch_warned = False
        self.anchor = None
        self.head0 = None
        self.angles = None  # latest filtered (yaw, pitch)
        self.slow = False  # slow-gain key held
        self.pos = None
        self.events = 0
        self.writer = None
        self.pending_clicks: list = []
        # Recent (frame, t, seq) so frames from before an event can still be saved.
        self.ring = collections.deque(maxlen=max(2, int(rc["context_before_s"] * 60)))
        self.marks: list[float] = []  # event times whose preceding frames are still to be saved
        self.save_until = -1.0
        self.moved_by_other = False  # another device moved the pointer since we last did

    # ------------------------------------------------------------ logging

    def _writer(self):
        if self.writer is None and self.cfg["run"]["log_usage"]:
            self.writer = record.SessionWriter(
                "usage",
                {
                    "monitors": [m.to_json() for m in self.mons],
                    "profile_key": profile_key(self.mons),
                    "camera": {"width": self.cam.width, "height": self.cam.height, "K": self.cam.K.tolist()},
                    "config": self.cfg,
                },
                self.cfg["collect"]["jpeg_quality"],
                async_frames=True,
            )
            self.writer.event({"type": "session", "t": time.monotonic(), "wall": time.time()})
            print(f"logging usage to {self.writer.dir}")
        return self.writer

    def _target(self, x: float, y: float, index: int, source: str) -> dict:
        _, _, m = clamp_desktop(self.mons, x, y)
        sc = next(s for s in self.screens if s.id == m.id)
        return {"monitor": m.id, "px": [x, y], "mm": list(sc.px_to_mm(x, y)), "index": index, "source": source}

    def _event(self, rec: dict):
        """Log a raw event and save the frames around it. Lock held."""
        if self._writer():
            self.writer.event(rec)
            self.marks.append(rec["t"])

    def _move(self, x: float, y: float):
        self.pointer.move(x, y)
        self.moved_by_other = False

    # ------------------------------------------------------------ input callbacks

    def on_key(self, down: bool, t: float):
        with self.lock:
            self._event({"type": "key", "t": t, "key": "aim", "down": down, "state": self.state})
            if down and self.state == "idle":
                self.state, self.latch_t, self.latch, self.latch_warned = "latching", t, [], False
                print("hotkey down: reading gaze")
            elif not down and self.state != "idle":
                self._release()

    def on_reaim(self, down: bool, t: float):
        """Jump to the gaze again without letting go of the hotkey. Mouse buttons
        held meanwhile stay held, so this carries a drag to its destination."""
        with self.lock:
            self._event({"type": "key", "t": t, "key": "reaim", "down": down, "state": self.state})
            if down and self.state != "idle":
                self._log_segment()
                self.state, self.latch_t, self.latch, self.latch_warned = "latching", t, [], False
                print("re-aim: reading gaze")

    def on_slow(self, down: bool, t: float):
        """Hold for the low head gain. Re-anchors at the current cursor so the
        cursor doesn't jump when the gain changes."""
        with self.lock:
            self._event({"type": "key", "t": t, "key": "slow", "down": down, "state": self.state})
            self.slow = down
            if self.state == "held" and self.angles:
                self.anchor, self.head0 = self.pos, self.angles

    def on_pointer(self, rec: dict):
        """Raw pointing-device event from the input thread (io.PointerDecoder)."""
        with self.lock:
            if rec["type"] in ("motion", "touch"):
                self.moved_by_other = True
            if rec["type"] != "button":
                if self._writer():
                    self.writer.event(rec)
                return
            rec.update(cursor=list(self.pos) if self.pos else None,
                       cursor_trusted=self.pos is not None and not self.moved_by_other, state=self.state)
            actual = self.pointer.position()  # X11 only; under the lock since Xlib isn't thread-safe
            if actual:
                rec["cursor_actual"] = list(actual)
            self._event(rec)

    def on_click(self, t: float):
        pos = self.pointer.position()
        item = self.cam.at(t - self.cfg["run"]["click_lookback_ms"] / 1000)
        if pos and item:
            with self.lock:
                self.pending_clicks.append((item, pos))

    def _release(self):
        """Called with lock held."""
        self._log_segment()
        if self.state == "held":
            print(f"hotkey up: cursor at ({self.pos[0]:.0f}, {self.pos[1]:.0f})")
        else:
            print("hotkey up before a gaze fix; cursor not moved")
        self.state = "idle"
        self.latch = []

    def _log_segment(self):
        """Label the latch frames with where the cursor ended up. Lock held."""
        if self.state == "held" and self.latch and self._writer():
            self.events += 1
            tgt = self._target(*self.pos, self.events, "release")
            extra = {"anchor_px": list(self.anchor), "event": self.events}
            for frame, obs, _hit in self.latch:
                self.writer.add(frame, obs.seq, obs.t, tgt, obs.to_json(), **extra)

    def _latch_stalled(self, t: float, why: str):
        """Called with lock held: say once per press why the jump hasn't happened yet."""
        if self.state == "latching" and not self.latch_warned and t - self.latch_t >= self.cfg["run"]["latch_ms"] / 1000:
            self.latch_warned = True
            print(f"still waiting to jump: {why}")

    # ------------------------------------------------------------ main loop

    def step(self, frame, t, seq):
        obs = self.tracker.process(frame, t, seq)
        if obs is None:
            with self.lock:
                self._latch_stalled(t, "face not detected")
                self._record_frame(frame, t, seq, None, None)
            return
        yaw = self.f_yaw(obs.head_yaw, t)
        pitch = self.f_pitch(obs.head_pitch, t)
        rc = self.cfg["run"]
        hit = gaze_px(self.A, self.screens, obs)
        with self.lock:
            self._step_locked(frame, t, seq, obs, hit, yaw, pitch, rc)
            self._record_frame(frame, t, seq, obs, hit)

    def _step_locked(self, frame, t, seq, obs, hit, yaw, pitch, rc):
        self.angles = (yaw, pitch)
        if self.state == "latching":
            if hit:
                self.latch.append((frame, obs, hit))
            else:
                self._latch_stalled(t, "gaze misses every screen")
            if t - self.latch_t >= rc["latch_ms"] / 1000 and self.latch:
                xs = np.array([[h[1], h[2]] for _, _, h in self.latch])
                self.anchor = tuple(np.median(xs, axis=0))
                self.head0 = (yaw, pitch)
                self.state = "held"
                x, y, m = clamp_desktop(self.mons, *self.anchor)
                self.pos = (x, y)
                self._move(*self.pos)
                self._event({"type": "jump", "t": t, "anchor": [x, y], "n": len(self.latch)})
                print(f"jump to ({x:.0f}, {y:.0f}) on {m.name}, median of {len(self.latch)} gaze frames")
        elif self.state == "held":
            g = rc["gain_slow_px_per_deg"] if self.slow else rc["gain_px_per_deg"]
            x = self.anchor[0] + g * (yaw - self.head0[0])
            y = self.anchor[1] - g * (pitch - self.head0[1])
            self.pos = clamp_desktop(self.mons, x, y)[:2]
            self._move(*self.pos)

    def _record_frame(self, frame, t, seq, obs, hit):
        """Log this frame to events.jsonl, saving its image during holds and
        around events (including frames from just before them). Lock held."""
        if not self._writer():
            return
        rc = self.cfg["run"]
        for tm in self.marks:
            for f, ft, fs in self.ring:
                if ft >= tm - rc["context_before_s"]:
                    self.writer.save_frame(f, fs)
            self.save_until = max(self.save_until, tm + rc["context_after_s"])
        self.marks = []
        self.ring.append((frame, t, seq))
        if t <= self.save_until or (self.state != "idle" and rc["save_hold_frames"]):
            self.writer.save_frame(frame, seq)
        self.writer.event({
            "type": "frame", "t": t, "seq": seq, "state": self.state,
            "obs": obs.to_json() if obs else None,
            "gaze": [hit[1], hit[2]] if hit else None,
            "cursor": list(self.pos) if self.state == "held" else None,
        })

    def run(self):
        self._writer()
        self.listener.start()
        kbds = ", ".join(d.name for d in self.listener.kbds)
        rc = self.cfg["run"]
        reaim = f", {rc['reaim_key']} to re-aim" if rc["reaim_key"] else ""
        print(f"ready: hold {rc['hotkey']} to point{reaim}; listening on {kbds} (Ctrl-C to quit)")
        seq = 0
        n, t_rate = 0, time.monotonic()
        try:
            while True:
                try:
                    frame, t, seq = self.cam.next(seq)
                except TimeoutError:
                    continue
                self.step(frame, t, seq)
                self._drain_clicks()
                n += 1
                if time.monotonic() - t_rate > 30:
                    print(f"{n / (time.monotonic() - t_rate):.1f} fps")
                    n, t_rate = 0, time.monotonic()
        except KeyboardInterrupt:
            pass
        finally:
            self.listener.stop()
            self.cam.close()
            self.tracker.close()
            self.pointer.close()
            if self.writer:
                self.writer.close()

    def _drain_clicks(self):
        with self.lock:
            clicks, self.pending_clicks = self.pending_clicks, []
        for (frame, t, seq), (x, y) in clicks:
            # Frames from the past can't go through the VIDEO-mode tracker; store
            # them unprocessed and fill `obs` later with `gazepoint reprocess`.
            if self._writer():
                self.events += 1
                self.writer.add(frame, seq, t, self._target(x, y, self.events, "click"), None, event=self.events)
