"""Guided data collection: fixation targets on every monitor, webcam frames labelled
with where the user was looking.

One fullscreen window per monitor. Each target shrinks over settle_ms (the user
saccades to it and settles), then holds for capture_ms while frames are recorded.
Every camera frame of a capture is saved and queued; perception runs on its own
thread in capture order, so a gaze net slower than the camera falls behind during
captures and catches up while the next target settles (and at the end). Once a
capture's frames are all processed, if the raw gaze doesn't look like a fixation the
presentation is listed in session.json's rejected_targets (frames kept, calibration
skips them) and the point is queued again at the end.
Space pauses and resumes; a capture interrupted by a pause is rejected the same way
and its point shown again straight after resuming. A bar along the bottom of every
screen shows progress.
After the last target the session is calibrated in place and the windows switch to
a live gaze view (G skips straight there using the saved calibration).
"""

from __future__ import annotations

import collections
import random
import shutil
import threading
import traceback
import time
import tkinter as tk

import cv2
import numpy as np

from . import calib, record
from .camera import Camera
from .monitors import Monitor, detect, profile_key
from .run import clamp_desktop, gaze_px
from .tracker import Tracker

# Queued capture frames beyond this many are dropped from memory (~2.8 MB each at
# 720p) and read back from their saved JPEG when their turn comes.
BACKLOG_IN_MEMORY = 64


def grid(n: int) -> list[tuple[float, float]]:
    """Target positions as fractions of the monitor."""
    lo, hi = 0.08, 0.92
    if n <= 5:
        return [(lo, lo), (hi, lo), (0.5, 0.5), (lo, hi), (hi, hi)][:n]
    k = int(np.ceil(np.sqrt(n)))
    pts = [(lo + (hi - lo) * i / (k - 1), lo + (hi - lo) * j / (k - 1)) for j in range(k) for i in range(k)]
    return pts[:n]


def fixation_problem(gaze: list, cc: dict) -> str | None:
    """Why one target's raw (yaw, pitch) samples don't look like a steady fixation, or None."""
    if len(gaze) < cc["fixation_min_frames"]:
        return f"only {len(gaze)} face frames"
    g = np.asarray(gaze, dtype=np.float64)
    spread = np.percentile(np.linalg.norm(g - np.median(g, axis=0), axis=1), 75)
    if spread > cc["fixation_spread_deg"]:
        return f"gaze scattered ({spread:.1f} deg)"
    h = len(g) // 2
    drift = np.linalg.norm(np.median(g[:h], axis=0) - np.median(g[h:], axis=0))
    if drift > cc["fixation_drift_deg"]:
        return f"gaze moved during capture ({drift:.1f} deg)"
    return None


def choose_mode(mons: list[Monitor], cfg: dict) -> str:
    if calib.load_user(cfg["gaze"]["model"]) is None:
        return "full"
    if all(m.builtin for m in mons):
        return "quick"
    return "quick" if calib.load_profile(profile_key(mons)) else "full"


def hide_cursor(window_ids: list[int]):
    """Blank the pointer over the given X windows. This goes through python-xlib's own
    connection because any Tk cursor (even cursor="none") aborts uv's Python: its Tk
    has a static libX11 that dlopens the system libXcursor, which then drives the
    same Display through the system libX11 ("xcb_xlib_unknown_seq_number").
    Returns the connection, which must stay open for the cursor to persist."""
    try:
        from Xlib import display

        d = display.Display()
        pm = d.screen().root.create_pixmap(1, 1, 1)
        gc = pm.create_gc(foreground=0, background=0)
        pm.fill_rectangle(gc, 0, 0, 1, 1)
        blank = pm.create_cursor(pm, (0, 0, 0), (0, 0, 0), 0, 0)
        for wid in window_ids:
            d.create_resource_object("window", wid).change_attributes(cursor=blank)
        d.sync()
        return d
    except Exception as e:  # cosmetic only
        print(f"WARNING: could not hide the cursor: {e}")
        return None


def summarize(report: dict) -> str:
    """Calibrate report -> one status line: median gaze error on each screen."""
    errs = [f"{name.split(' [')[0]} {e['deg']['median']}°" for name, e in report["eval"].items() if e["deg"]]
    line = "calibrated, median error: " + ", ".join(errs)
    if "warning" in report:
        line += "\nWARNING: " + report["warning"]
    return line


class Collector:
    def __init__(self, cfg: dict, mode: str, rounds: int, points: int | None, do_calibrate: bool = True):
        self.cfg = cfg
        self.do_calibrate = do_calibrate
        self.mons = detect()
        self.mode = choose_mode(self.mons, cfg) if mode == "auto" else mode
        cc = cfg["collect"]
        n = points or (cc["points_full"] if self.mode == "full" else cc["points_quick"])
        self.targets = []
        for r in range(rounds):
            order = self.mons[:]
            random.shuffle(order)
            for m in order:
                pts = grid(n)
                random.shuffle(pts)
                self.targets += [(m, fx, fy) for fx, fy in pts]
        self.cam = Camera(cfg)
        self.tracker = Tracker(cfg, self.cam.K)
        self.writer = record.SessionWriter(
            "collect",
            {
                "mode": self.mode,
                "monitors": [m.to_json() for m in self.mons],
                "profile_key": profile_key(self.mons),
                "camera": {
                    "width": self.cam.width,
                    "height": self.cam.height,
                    "fps": self.cam.fps,
                    "K": self.cam.K.tolist(),
                },
                "config": cfg,
            },
            cc["jpeg_quality"],
        )
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)  # capture -> process: new frame
        self.current: dict | None = None  # target being captured
        self.face_ok = False
        self.stop = False
        self.aborted = False
        self.threads: list[threading.Thread] = []
        # (frame or None if only on disk, t, seq, target), captured but not yet processed
        self.backlog: collections.deque = collections.deque()
        self.unprocessed: collections.Counter = collections.Counter()  # target index -> frames pending
        self.captured = 0
        self.processed = 0
        self.idle: tuple | None = None  # newest (frame, t, seq) outside a capture
        self.gaze: dict[int, list] = {}  # target index -> raw (yaw, pitch) per face frame
        self.to_judge: dict[int, tuple] = {}  # index -> (monitor, fx, fy): capture over, not judged yet
        self.pos = 0  # presentation on screen, or the one to show after a pause
        self.paused = False
        self.gen = 0  # bumped to cancel the running presentation's animation
        self.rejected: list[int] = []
        self.retries: collections.Counter = collections.Counter()  # (monitor id, fx, fy) -> re-shows
        self.phase = "intro"  # intro -> targets -> calibrating -> view
        self.calib_thread: threading.Thread | None = None
        self.calib_result: dict | Exception | None = None
        self.view: tuple | None = None  # (A, screens) once the gaze view is live
        self.view_msg = ""
        self.hits: collections.deque = collections.deque(maxlen=120)  # (t, desktop x, y)

    # ------------------------------------------------------------ worker

    def capture(self):
        """Camera -> backlog. During a capture every frame is saved and queued; outside
        one only the newest is kept (face check, live gaze view)."""
        seq = 0
        while not self.stop:
            try:
                frame, t, seq = self.cam.next(seq)
            except TimeoutError:
                continue
            with self.wake:
                target = self.current
                if target is None:
                    self.idle = (frame, t, seq)
                    self.wake.notify()
                    continue
                # Counted before it's queued so the target isn't judged in between.
                self.unprocessed[target["index"]] += 1
                keep = len(self.backlog) < BACKLOG_IN_MEMORY
            self.writer.save_frame(frame, seq)
            with self.wake:
                self.backlog.append((frame if keep else None, t, seq, target))
                self.captured += 1
                self.wake.notify()

    def process(self):
        """Perception in capture order, as fast as the gaze net allows: the backlog
        first, else the newest idle frame. On stop, whatever is still queued is
        recorded with obs null (`gazepoint reprocess` fills it in)."""
        last = 0
        while True:
            with self.wake:
                self.wake.wait_for(
                    lambda: self.stop or self.backlog or (self.idle is not None and self.idle[2] > last)
                )
                if self.stop:
                    rest = list(self.backlog)
                    self.backlog.clear()
                    break
                if self.backlog:
                    frame, t, seq, target = self.backlog.popleft()
                else:
                    (frame, t, seq), target = self.idle, None
            last = seq
            if frame is None:
                frame = cv2.imread(str(self.writer.frame_path(seq)))
            obs = self.tracker.process(frame, t, seq, keep_landmarks=target is not None)
            self.face_ok = obs is not None
            if target is not None:
                self.writer.add(frame, seq, t, target, obs.to_json() if obs else None)
                with self.lock:
                    if obs:
                        self.gaze.setdefault(target["index"], []).append(obs.gaze_raw)
                    self.unprocessed[target["index"]] -= 1
                    self.processed += 1
            view = self.view
            if view and obs:
                hit = gaze_px(*view, obs)
                if hit:
                    with self.lock:
                        self.hits.append((t, hit[1], hit[2]))
        for frame, t, seq, target in rest:
            self.writer.add(frame, seq, t, target, None)

    def shutdown(self):
        with self.wake:
            self.stop = True
            self.wake.notify_all()
        for th in self.threads:
            th.join()

    # ------------------------------------------------------------ UI

    def run(self) -> str:
        cc = self.cfg["collect"]
        root = tk.Tk()
        root.withdraw()
        self.wins = {}
        geoms = {}
        for m in self.mons:
            w = tk.Toplevel(root, background=cc["background"])
            w.geometry(f"{m.w}x{m.h}+{m.x}+{m.y}")
            w.update_idletasks()
            w.attributes("-fullscreen", True)
            c = tk.Canvas(w, background=cc["background"], highlightthickness=0)
            c.pack(fill="both", expand=True)
            for seq in ("<space>", "<Escape>", "q", "g", "G"):
                w.bind(seq, self._key)
            self.wins[m.id] = (w, c, m)
        root.update()
        time.sleep(0.3)
        root.update()
        for mid, (w, c, m) in self.wins.items():
            geoms[mid] = [w.winfo_rootx(), w.winfo_rooty(), w.winfo_width(), w.winfo_height()]
            if geoms[mid] != [m.x, m.y, m.w, m.h]:
                print(f"WARNING: window for {m.name} landed at {geoms[mid]}, expected {[m.x, m.y, m.w, m.h]}")
        self.writer.update_meta(window_geometry=geoms)
        self._xcursor = hide_cursor([c.winfo_id() for _, c, _ in self.wins.values()])
        self._builtin()[0].focus_force()
        self.root = root
        for fn in (self.capture, self.process):
            th = threading.Thread(target=fn, name=fn.__name__, daemon=True)
            th.start()
            self.threads.append(th)
        self._intro()
        root.mainloop()
        self.shutdown()
        if self._xcursor:
            self._xcursor.close()
        return self.writer.id

    def _builtin(self):
        return next(v for v in self.wins.values() if v[2].builtin)

    def _key(self, ev):
        if ev.keysym in ("Escape", "q"):
            if self.phase in ("intro", "targets"):
                print("aborted")
                self.aborted = True
            self.root.quit()
        elif ev.keysym == "space" and self.phase == "intro":
            self.phase = "targets"
            self._show(0)
        elif ev.keysym == "space" and self.phase == "targets":
            self._toggle_pause()
        elif ev.keysym in ("g", "G") and self.phase in ("intro", "targets"):
            print("calibration skipped")
            with self.lock:
                self.current = None
            self._view("calibration skipped, using the saved one")

    def _clear(self):
        for _, c, _ in self.wins.values():
            c.delete("all")

    def _intro(self):
        if self.phase != "intro":
            return
        self._clear()
        _, c, m = self._builtin()
        n_ext = sum(1 for x in self.mons if not x.builtin)
        msg = (
            f"gazepoint collect  -  {self.mode} calibration, {len(self.targets)} targets, "
            f"{len(self.mons)} screen(s) ({n_ext} external)\n\n"
            "Look at each dot until it disappears.\n"
            "Keep your head natural; it's fine (good, even) to shift around between dots.\n"
            "Afterwards you'll see where it thinks you're looking.\n\n"
            f"Face: {'OK' if self.face_ok else 'NOT DETECTED'}\n\n"
            "Space = start, then pause/resume      G = skip calibration, show live gaze      Esc = abort"
        )
        c.create_text(m.w // 2, m.h // 2, text=msg, fill="white", font=("Sans", 16), justify="center")
        self.root.after(250, self._intro)

    def _show(self, i: int):
        if self.phase != "targets" or self.paused:
            return
        self.pos = i
        self._judge_ready()
        if i >= len(self.targets):
            self._finish()
            return
        m, fx, fy = self.targets[i]
        _, c, _ = self.wins[m.id]
        x, y = fx * m.w, fy * m.h
        target = {
            "monitor": m.id,
            "px": [m.x + x, m.y + y],
            "mm": [fx * m.width_mm, fy * m.height_mm],
            "index": i,
        }
        cc = self.cfg["collect"]
        t0 = time.monotonic()
        settle, capture = cc["settle_ms"] / 1000, cc["capture_ms"] / 1000
        gen = self.gen

        def frame():
            if self.phase != "targets" or gen != self.gen:
                return
            el = time.monotonic() - t0
            self._clear()
            if el < settle:
                r = 6 + 34 * (1 - el / settle)
                c.create_oval(x - r, y - r, x + r, y + r, outline="white", width=3)
                c.create_oval(x - 4, y - 4, x + 4, y + 4, fill="red", outline="")
            elif el < settle + capture:
                with self.lock:
                    self.current = target
                c.create_oval(x - 6, y - 6, x + 6, y + 6, fill="red", outline="")
                c.create_oval(x - 1.5, y - 1.5, x + 1.5, y + 1.5, fill="white", outline="")
            else:
                with self.lock:
                    self.current = None
                    self.to_judge[i] = (m, fx, fy)
                self.root.after(1, self._show, i + 1)
                return
            if not self.face_ok:
                c.create_text(m.w // 2, 30, text="face not detected", fill="orange", font=("Sans", 14))
            self._target_progress()
            self.root.after(16, frame)

        frame()

    def _toggle_pause(self):
        if self.paused:
            self.paused = False
            self._show(self.pos)
            return
        self.paused = True
        self.gen += 1
        with self.lock:
            target, self.current = self.current, None
        if target is not None:  # mid-capture: reject it and show the same point next
            i = target["index"]
            self.rejected.append(i)
            self.writer.update_meta(rejected_targets=self.rejected)
            self.targets.insert(i + 1, self.targets[i])
            self.pos = i + 1
            print(f"target {i}: interrupted by pause; showing it again")
        self._draw_paused()

    def _draw_paused(self):
        if not self.paused or self.phase != "targets":
            return
        self._clear()
        _, c, m = self._builtin()
        msg = (f"Paused\n\nFace: {'OK' if self.face_ok else 'NOT DETECTED'}\n\n"
               "Space = resume      G = skip calibration, show live gaze      Esc = abort")
        c.create_text(m.w // 2, m.h // 2, text=msg, fill="white", font=("Sans", 16), justify="center")
        self._target_progress()
        self.root.after(250, self._draw_paused)

    def _finish(self):
        """After the last target: wait for the backlog to drain and every capture to be
        judged (which may queue retries), then calibrate."""
        if self.phase != "targets" or self.paused:
            return
        self._judge_ready()
        if self.pos < len(self.targets):  # retries were queued
            self._show(self.pos)
            return
        if self.to_judge:
            self._clear()
            _, c, m = self._builtin()
            with self.lock:
                left, done, total = len(self.backlog), self.processed, self.captured
            c.create_text(m.w // 2, m.h // 2, text=f"Processing recorded frames: {left} left",
                          fill="white", font=("Sans", 20))
            self._progress(done, total, f"{done}/{total} frames")
            self.root.after(100, self._finish)
            return
        if self.do_calibrate:
            self._calibrate()
        else:
            self.root.quit()

    def _progress(self, done: int, total: int, label: str):
        """A thin bar along the bottom of every screen."""
        h = 6
        for _, c, m in self.wins.values():
            c.create_rectangle(0, m.h - h, m.w, m.h, fill="#484848", outline="")
            c.create_rectangle(0, m.h - h, m.w * done / max(total, 1), m.h, fill="#6fa8dc", outline="")
            c.create_text(m.w - 12, m.h - h - 4, text=label, fill="#a0a0a0", font=("Sans", 11), anchor="se")

    def _target_progress(self):
        """Presentations done out of those queued (retries add to the total). Also
        shows how far perception has fallen behind when it's slower than the camera."""
        label = f"{self.pos}/{len(self.targets)}"
        left = len(self.backlog)
        if left > 30:
            label += f"  ({left} frames to process)"
        self._progress(self.pos, len(self.targets), label)

    def _judge_ready(self):
        """Judge the finished captures whose frames have all been processed."""
        with self.lock:
            ready = sorted(i for i in self.to_judge if self.unprocessed[i] == 0)
        for i in ready:
            self._judge(i, *self.to_judge.pop(i))

    def _judge(self, i: int, m: Monitor, fx: float, fy: float):
        with self.lock:
            gaze = list(self.gaze.get(i, []))
        why = fixation_problem(gaze, self.cfg["collect"])
        if why is None:
            return
        self.rejected.append(i)
        self.writer.update_meta(rejected_targets=self.rejected)
        key = (m.id, fx, fy)
        if self.retries[key] < self.cfg["collect"]["fixation_retries"]:
            self.retries[key] += 1
            self.targets.append((m, fx, fy))
            print(f"target {i}: {why}; showing it again later")
        else:
            print(f"target {i}: {why}; out of retries, skipping")

    # ------------------------------------------------------------ calibrate + gaze view

    def _calibrate(self):
        from .cli import run_calibrate

        self.phase = "calibrating"
        self._clear()
        _, c, m = self._builtin()
        c.create_text(m.w // 2, m.h // 2, text="Calibrating...", fill="white", font=("Sans", 20))

        def fit():
            try:
                self.calib_result = run_calibrate(self.cfg, self.writer.id, self.mode, refit_user=False)
            except Exception as e:  # noqa: BLE001 - reported in the UI
                traceback.print_exc()
                self.calib_result = e

        self.calib_thread = threading.Thread(target=fit, name="calibrate")
        self.calib_thread.start()
        self._await_calibration()

    def _await_calibration(self):
        if self.calib_thread.is_alive():
            self.root.after(100, self._await_calibration)
            return
        r = self.calib_result
        if isinstance(r, Exception):
            self._view(f"calibration FAILED: {r}\nshowing the saved calibration instead")
        else:
            self._view(summarize(r))

    def _view(self, msg: str):
        try:
            self.view = calib.load_runtime(self.mons, self.cfg)
        except calib.NotCalibrated as e:
            msg += f"\nno usable calibration: {e}"
        self.phase = "view"
        self.view_msg = msg
        self._draw_view()

    def _draw_view(self):
        """Blue trail = raw gaze hits over the last 0.5 s. White ring = their median over
        latch_ms, i.e. where `gazepoint run` would jump if you pressed the hotkey now."""
        now = time.monotonic()
        with self.lock:
            hits = [h for h in self.hits if now - h[0] < 0.5]
        self._clear()
        for _, c, m in self.wins.values():
            for _, x, y in hits:
                lx, ly = x - m.x, y - m.y
                if 0 <= lx < m.w and 0 <= ly < m.h:
                    c.create_oval(lx - 3, ly - 3, lx + 3, ly + 3, fill="#6fa8dc", outline="")
        latch = [(x, y) for t, x, y in hits if now - t < self.cfg["run"]["latch_ms"] / 1000]
        if latch:
            x, y, m = clamp_desktop(self.mons, *np.median(latch, axis=0))
            _, c, _ = self.wins[m.id]
            lx, ly = x - m.x, y - m.y
            c.create_oval(lx - 16, ly - 16, lx + 16, ly + 16, outline="white", width=3)
            c.create_oval(lx - 2, ly - 2, lx + 2, ly + 2, fill="white", outline="")
        _, c, m = self._builtin()
        status = self.view_msg
        if self.view and not self.face_ok:
            status += "\nface not detected"
        c.create_text(m.w // 2, 30, text=status + "\n\nEsc = quit", fill="white", font=("Sans", 14),
                      justify="center", anchor="n")
        self.root.after(16, self._draw_view)


def collect(cfg: dict, mode: str = "auto", rounds: int = 1, points: int | None = None, do_calibrate: bool = True):
    col = Collector(cfg, mode, rounds, points, do_calibrate)
    print(f"session {col.writer.id}: mode={col.mode}, {len(col.targets)} targets")
    try:
        col.run()
    finally:
        col.shutdown()
        if col.calib_thread and col.calib_thread.is_alive():
            print("waiting for calibration to finish...")
            col.calib_thread.join()  # never leave calibration files half-written
        col.writer.close()
        col.cam.close()
        col.tracker.close()
    if col.writer.count == 0:
        shutil.rmtree(col.writer.dir)
        print("no frames recorded; session discarded")
        return None
    shown = min(col.pos, len(col.targets))
    good = shown - len([i for i in col.rejected if i < shown])
    print(f"recorded {col.writer.count} frames, {good}/{shown} target presentations kept"
          + (f" ({len(col.rejected)} rejected: not fixated or paused)" if col.rejected else ""))
    print(f"data: {col.writer.dir}")
    return col.writer.id
