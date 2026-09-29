"""Live mode: hold the hotkey -> cursor jumps to gaze, head motion refines.

State machine (driven by camera frames and gaze results):

  idle --key down--> latching: gather gaze hits for latch_ms          (aim_mode jump)
  idle --key down--> held, anchored at the current cursor, no jump    (aim_mode anchor)
  latching --------> held: warp to median hit (the anchor), remember head pose
  held: cursor = anchor + gain * (head delta)   (head angles One-Euro filtered)
  held --key up----> idle: log (latch frames, anchor, final cursor) as a label
  held/latching --reaim key down--> latching on the gaze hits from the latch_ms
      *before* the press (jumps at once if there are any); logs the segment so far
  held --reaim key still down after track_after_ms--> tracking: the cursor
      follows the smoothed gaze (GazeSmoother)
  tracking --reaim key up--> held, anchored where tracking left the cursor
  held --scroll key down--> cursor frozen; head displacement controls scroll
  scrolling --scroll key up--> held, head motion re-anchored without a jump

In anchor mode on Wayland the cursor position can't be read, so unless
gazepoint put it where it is, head motion moves it by relative steps until the
first jump. With run.gaze_thread the gaze net runs on a worker thread
(GazeWorker) on the newest frame, and gaze hits arrive at the net's own rate.

The final cursor position is usually where the user wanted to be, so each use
produces a labelled sample (samples.jsonl) for later recalibration/training.
It isn't always: holds get abandoned or retried. So usage sessions also keep a
raw stream in events.jsonl, from which labels can be rederived offline:

  session  t, wall            clock alignment (wall = time.time() at monotonic t)
  frame    t, seq, state, obs, gaze [x, y] (current calibration), cursor
           (obs has gaze_raw None when the net didn't run on this frame here)
  gaze     t, seq, obs, gaze   (gaze_thread only: net results, t = capture time)
  key      t, key (aim|reaim|slow|scroll), down, state
  anchor   t, cursor (None if unknown: moving relatively)   (anchor mode)
  jump     t, anchor, n (gaze frames in the median)
  track    t, down (tracking started/stopped), cursor
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
import traceback

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


class GazeSmoother:
    """Smooths the gaze hits the cursor follows while tracking (run.track_*)."""

    def __init__(self, rc: dict, x: float, y: float, t: float):
        self.kind = rc["track_filter"]
        if self.kind not in ("one_euro", "ema", "median", "none"):
            raise ValueError(f"run.track_filter must be one_euro, ema, median, or none, not {self.kind!r}")
        self.tau = rc["track_tau_ms"] / 1000
        self.window = rc["track_window_ms"] / 1000
        self.f = [OneEuro(rc["track_min_cutoff"], rc["track_beta"]) for _ in range(2)]
        self.hist: collections.deque = collections.deque()
        self.pos, self.t = (x, y), t
        for f, v in zip(self.f, self.pos):  # start from where the cursor is
            f(v, t)

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        if self.kind == "one_euro":
            self.pos = (self.f[0](x, t), self.f[1](y, t))
        elif self.kind == "ema":
            a = 1 - np.exp(-max(t - self.t, 0.0) / self.tau) if self.tau > 0 else 1.0
            self.pos = (self.pos[0] + a * (x - self.pos[0]), self.pos[1] + a * (y - self.pos[1]))
        elif self.kind == "median":
            self.hist.append((t, x, y))
            while self.hist[0][0] < t - self.window:
                self.hist.popleft()
            self.pos = tuple(np.median(np.array([h[1:] for h in self.hist]), axis=0))
        else:
            self.pos = (x, y)
        self.t = t
        return self.pos


class HeadScroller:
    """Integrate head displacement from a neutral pose into wheel detents."""

    def __init__(self, rc: dict, angles: tuple[float, float], t: float):
        self.rc, self.neutral, self.t = rc, angles, t
        self.velocity = np.zeros(2, dtype=float)  # horizontal, vertical detents/s

    def _speed(self, delta: float, gain: float) -> float:
        d = max(abs(delta) - self.rc["scroll_deadzone_deg"], 0.0)
        speed = gain * d ** self.rc["scroll_curve"]
        return np.sign(delta) * min(speed, self.rc["scroll_max_steps_per_s"])

    def __call__(self, angles: tuple[float, float], t: float, slow: bool = False):
        dt = min(max(t - self.t, 0.0), 0.1)
        yaw, pitch = angles
        target = np.array([
            self._speed(yaw - self.neutral[0], self.rc["scroll_horizontal_steps_per_s_per_deg"])
            if self.rc["scroll_horizontal"] else 0.0,
            self._speed(self.neutral[1] - pitch, self.rc["scroll_vertical_steps_per_s_per_deg"]),
        ])
        if slow:
            target *= self.rc["scroll_slow_factor"]
        tau = self.rc["scroll_tau_ms"] / 1000
        a = 1 - np.exp(-dt / tau) if tau > 0 else 1.0
        self.velocity += a * (target - self.velocity)
        self.t = t
        return tuple(self.velocity * dt)


class GazeWorker(threading.Thread):
    """Runs the gaze net on the newest submitted frame; older ones are dropped."""

    def __init__(self, tracker: Tracker, on_gaze):
        super().__init__(daemon=True, name="gaze")
        self.tracker, self.on_gaze = tracker, on_gaze
        self.cond = threading.Condition()
        self.slot = None
        self.stopped = False

    def submit(self, frame, pending):
        with self.cond:
            self.slot = (frame, pending)
            self.cond.notify()

    def stop(self):
        with self.cond:
            self.stopped = True
            self.cond.notify()

    def run(self):
        while True:
            with self.cond:
                while self.slot is None and not self.stopped:
                    self.cond.wait()
                if self.stopped:
                    return
                (frame, p), self.slot = self.slot, None
            try:
                self.on_gaze(frame, self.tracker.gaze(p))
            except Exception:
                traceback.print_exc()


class Live:
    def __init__(self, cfg: dict, log_clicks: bool | None = None):
        self.cfg = cfg
        rc = cfg["run"]
        self.mons = detect()
        self.A, self.screens = calib.load_runtime(self.mons, cfg)
        self.cam = Camera(cfg)
        self.tracker = Tracker(cfg, self.cam.K)
        self.worker = GazeWorker(self.tracker, self.on_gaze) if rc["gaze_thread"] else None
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
                                      # Anchor mode needs to know when another device moves the pointer.
                                      self.on_pointer if rc["log_usage"] or rc["aim_mode"] == "anchor" else None,
                                      rc["slow_key"], self.on_slow,
                                      rc["scroll_key"], self.on_scroll)

    def _init_state(self):
        rc = self.cfg["run"]
        self.state = "idle"
        self.latch_t = 0.0
        self.latch_from = 0.0  # gaze hits captured from here on count toward the jump
        self.preroll = False  # latch on hits from before the press: jump as soon as there are any
        self.latch: list = []  # (frame, obs, hit)
        self.latch_warned = False
        self.recent = collections.deque()  # (frame, obs, hit) gaze results, last ~latch_ms + 0.5 s
        self.anchor = None  # None while held = cursor position unknown, moving relatively
        self.rel_prev = None  # head angles at the last relative move
        self.head0 = None
        self.angles = None  # latest filtered (yaw, pitch)
        self.slow = False  # slow-gain key held
        self.scroller = None  # HeadScroller while the scroll key is held
        self.reaim_t = None  # when the re-aim key went down, while it's held
        self.smoother = None  # GazeSmoother while tracking
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

    def _start_latch(self, t: float, preroll: bool):
        """Lock held."""
        self.state, self.latch_t, self.latch, self.latch_warned = "latching", t, [], False
        self.preroll = preroll
        self.latch_from = t - self.cfg["run"]["latch_ms"] / 1000 if preroll else t
        self._try_jump(t)

    def on_key(self, down: bool, t: float):
        with self.lock:
            self._event({"type": "key", "t": t, "key": "aim", "down": down, "state": self.state})
            if down and self.state == "idle":
                if self.cfg["run"]["aim_mode"] == "anchor":
                    self._anchor_here(t)
                else:
                    self._start_latch(t, preroll=False)
                    print("hotkey down: reading gaze")
            elif not down and self.state != "idle":
                self._release()

    def _anchor_here(self, t: float):
        """Anchor mode: start moving the cursor from where it is. Lock held."""
        pos = self.pointer.position()
        if pos is None and self.pos is not None and not self.moved_by_other:
            pos = self.pos  # Wayland, but nothing else has moved it since we put it there
        self.state, self.latch = "held", []
        self.recent.clear()  # gaze from before the press predates this hold
        self.anchor = self.pos = tuple(pos) if pos else None
        self.head0 = self.rel_prev = self.angles
        self._event({"type": "anchor", "t": t, "cursor": list(pos) if pos else None})
        where = f"at ({pos[0]:.0f}, {pos[1]:.0f})" if pos else "(position unknown: moving relatively)"
        print(f"hotkey down: head moves the cursor {where}")

    def on_reaim(self, down: bool, t: float):
        """Jump to the gaze again without letting go of the hotkey. Mouse buttons
        held meanwhile stay held, so this carries a drag to its destination."""
        with self.lock:
            self._event({"type": "key", "t": t, "key": "reaim", "down": down, "state": self.state})
            if down and self.state != "idle":
                self._log_segment()
                print("re-aim: reading gaze")
                self.reaim_t = t
                self._start_latch(t, preroll=True)
            elif not down:
                self.reaim_t = None
                if self.state == "tracking":
                    self._stop_tracking(t)

    def _maybe_track(self, t: float):
        """Start following the gaze once the re-aim key has been held long enough. Lock held."""
        if (self.state == "held" and self.reaim_t is not None and self.anchor is not None
                and t - self.reaim_t >= self.cfg["run"]["track_after_ms"] / 1000):
            self.state = "tracking"
            self.smoother = GazeSmoother(self.cfg["run"], *self.pos, t)
            self._event({"type": "track", "t": t, "down": True, "cursor": list(self.pos)})
            print("tracking gaze")

    def _stop_tracking(self, t: float):
        """Back to head refinement from where tracking left the cursor. Lock held."""
        self.state, self.smoother = "held", None
        self.anchor, self.head0 = self.pos, self.angles
        self.latch = []  # the cursor no longer marks where the latch frames looked
        self._event({"type": "track", "t": t, "down": False, "cursor": list(self.pos)})
        print(f"tracking stopped at ({self.pos[0]:.0f}, {self.pos[1]:.0f})")

    def on_slow(self, down: bool, t: float):
        """Hold for the low head gain. Re-anchors at the current cursor so the
        cursor doesn't jump when the gain changes."""
        with self.lock:
            self._event({"type": "key", "t": t, "key": "slow", "down": down, "state": self.state})
            self.slow = down
            if self.state == "held" and self.angles and self.anchor is not None:
                self.anchor, self.head0 = self.pos, self.angles

    def on_scroll(self, down: bool, t: float):
        """Hold to freeze the cursor and control scroll velocity with head pose."""
        with self.lock:
            self._event({"type": "key", "t": t, "key": "scroll", "down": down, "state": self.state})
            if down and self.state == "held" and self.angles is not None and self.scroller is None:
                self.scroller = HeadScroller(self.cfg["run"], self.angles, t)
                print("head scroll on")
            elif not down and self.scroller is not None:
                self.scroller = None
                self.head0 = self.rel_prev = self.angles
                if self.anchor is not None:
                    self.anchor = self.pos
                print("head scroll off")

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
        if self.state in ("held", "tracking") and self.anchor is not None:
            print(f"hotkey up: cursor at ({self.pos[0]:.0f}, {self.pos[1]:.0f})")
        elif self.state == "held":
            print("hotkey up")
        else:
            print("hotkey up before a gaze fix; cursor not moved")
        self.state, self.smoother, self.scroller = "idle", None, None
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

    def _gaze_wanted(self) -> bool:
        return self.cfg["run"]["aim_mode"] != "anchor" or self.state != "idle"

    def step(self, frame, t, seq):
        p = self.tracker.head(frame, t, seq)
        if p is None:
            with self.lock:
                self._latch_stalled(t, "face not detected")
                self._record_frame(frame, t, seq, None, None)
            return
        yaw = self.f_yaw(p.obs.head_yaw, t)
        pitch = self.f_pitch(p.obs.head_pitch, t)
        obs, hit = p.obs, None
        want = self._gaze_wanted()
        if want and not self.worker:
            obs = self.tracker.gaze(p)
            hit = gaze_px(self.A, self.screens, obs)
        with self.lock:
            self.angles = (yaw, pitch)
            if obs.gaze_raw is not None:
                self._add_gaze(frame, obs, hit)
            self._maybe_track(t)
            if self.scroller is not None:
                self.pointer.scroll(*self.scroller(self.angles, t, self.slow))
            else:
                self._follow_head()
            self._record_frame(frame, t, seq, obs, hit)
        if want and self.worker:
            self.worker.submit(frame, p)

    def on_gaze(self, frame, obs):
        """A gaze result from the worker thread."""
        hit = gaze_px(self.A, self.screens, obs)
        with self.lock:
            if self._gaze_wanted():
                self._add_gaze(frame, obs, hit)
                if self._writer():
                    self.writer.event({"type": "gaze", "t": obs.t, "seq": obs.seq, "obs": obs.to_json(),
                                       "gaze": [hit[1], hit[2]] if hit else None})

    def _add_gaze(self, frame, obs, hit):
        """Lock held."""
        self.recent.append((frame, obs, hit))
        while self.recent and self.recent[0][1].t < obs.t - self.cfg["run"]["latch_ms"] / 1000 - 0.5:
            self.recent.popleft()
        if self.state == "latching":
            if not hit:
                self._latch_stalled(obs.t, "gaze misses every screen")
            self._try_jump(obs.t)
        self._maybe_track(obs.t)
        if self.state == "tracking" and self.scroller is None and hit:
            x, y = self.smoother(hit[1], hit[2], obs.t)
            self.pos = clamp_desktop(self.mons, x, y)[:2]
            self._move(*self.pos)

    def _try_jump(self, t: float):
        """Jump to the median gaze hit once the latch window is complete (or, for
        a preroll latch, as soon as it has any hits). Lock held."""
        rc = self.cfg["run"]
        hits = [r for r in self.recent if r[1].t >= self.latch_from and r[2]]
        if not hits or not (self.preroll or t - self.latch_t >= rc["latch_ms"] / 1000):
            return
        self.latch = hits
        xs = np.array([[h[1], h[2]] for _, _, h in hits])
        x, y, m = clamp_desktop(self.mons, *np.median(xs, axis=0))
        self.anchor = self.pos = (x, y)
        self.head0 = self.angles
        self.state = "held"
        self._move(x, y)
        self._event({"type": "jump", "t": t, "anchor": [x, y], "n": len(hits)})
        print(f"jump to ({x:.0f}, {y:.0f}) on {m.name}, median of {len(hits)} gaze frames")

    def _follow_head(self):
        """Held: cursor = anchor + gain * head delta (or relative steps while the
        cursor position is unknown). Lock held."""
        if self.state != "held" or self.angles is None:
            return
        rc = self.cfg["run"]
        g = rc["gain_slow_px_per_deg"] if self.slow else rc["gain_px_per_deg"]
        yaw, pitch = self.angles
        if self.head0 is None:
            self.head0 = self.rel_prev = self.angles
        if self.anchor is None:
            self.pointer.move_rel(g * (yaw - self.rel_prev[0]), -g * (pitch - self.rel_prev[1]))
            self.rel_prev = self.angles
            return
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
            "cursor": list(self.pos) if self.state in ("held", "tracking") and self.anchor is not None else None,
        })

    def run(self):
        self._writer()
        self.listener.start()
        if self.worker:
            self.worker.start()
        kbds = ", ".join(d.name for d in self.listener.kbds)
        rc = self.cfg["run"]
        if rc["aim_mode"] == "anchor":
            if not rc["reaim_key"]:
                print("WARNING: aim_mode = anchor but no reaim_key set: nothing will jump to the gaze")
            what = (f"hold {rc['hotkey']} to move the cursor with your head, tap {rc['reaim_key']} to jump to"
                    f" the gaze, hold it to follow the gaze, hold {rc['scroll_key']} to scroll with your head")
        else:
            reaim = f", tap {rc['reaim_key']} to re-aim, hold it to follow the gaze" if rc["reaim_key"] else ""
            what = f"hold {rc['hotkey']} to point{reaim}"
        print(f"ready: {what}; listening on {kbds} (Ctrl-C to quit)")
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
            if self.worker:
                self.worker.stop()
                self.worker.join()
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
