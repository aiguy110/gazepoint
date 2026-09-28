"""Guided data collection: fixation targets on every monitor, webcam frames labelled
with where the user was looking.

One fullscreen window per monitor. Each target shrinks over settle_ms (the user
saccades to it and settles), then holds for capture_ms while frames are recorded.
"""

from __future__ import annotations

import random
import threading
import time
import tkinter as tk

import numpy as np

from . import calib, record
from .camera import Camera
from .monitors import Monitor, detect, profile_key
from .tracker import Tracker


def grid(n: int) -> list[tuple[float, float]]:
    """Target positions as fractions of the monitor."""
    lo, hi = 0.08, 0.92
    if n <= 5:
        return [(lo, lo), (hi, lo), (0.5, 0.5), (lo, hi), (hi, hi)][:n]
    k = int(np.ceil(np.sqrt(n)))
    pts = [(lo + (hi - lo) * i / (k - 1), lo + (hi - lo) * j / (k - 1)) for j in range(k) for i in range(k)]
    return pts[:n]


def choose_mode(mons: list[Monitor]) -> str:
    if calib.load_user() is None:
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


class Collector:
    def __init__(self, cfg: dict, mode: str, rounds: int, points: int | None):
        self.cfg = cfg
        self.mons = detect()
        self.mode = choose_mode(self.mons) if mode == "auto" else mode
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
        self.current: dict | None = None  # target being captured
        self.face_ok = False
        self.stop = False
        self.aborted = False
        self.per_target: dict[int, int] = {}

    # ------------------------------------------------------------ worker

    def worker(self):
        seq = 0
        while not self.stop:
            try:
                frame, t, seq = self.cam.next(seq)
            except TimeoutError:
                continue
            with self.lock:
                target = self.current
            obs = self.tracker.process(frame, t, seq, keep_landmarks=target is not None)
            self.face_ok = obs is not None
            if target is not None:
                self.writer.add(frame, seq, t, target, obs.to_json() if obs else None)
                if obs:
                    self.per_target[target["index"]] = self.per_target.get(target["index"], 0) + 1

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
            for seq in ("<space>", "<Escape>", "q"):
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
        builtin = next(v for v in self.wins.values() if v[2].builtin)
        builtin[0].focus_force()
        self.root = root
        self.started = False
        threading.Thread(target=self.worker, daemon=True).start()
        self._intro()
        root.mainloop()
        self.stop = True
        if self._xcursor:
            self._xcursor.close()
        return self.writer.id

    def _key(self, ev):
        if ev.keysym in ("Escape", "q"):
            print("aborted")
            self.aborted = True
            self.root.quit()
        elif ev.keysym == "space" and not self.started:
            self.started = True
            self._show(0)

    def _clear(self):
        for _, c, _ in self.wins.values():
            c.delete("all")

    def _intro(self):
        if self.started:
            return
        self._clear()
        _, c, m = next(v for v in self.wins.values() if v[2].builtin)
        n_ext = sum(1 for x in self.mons if not x.builtin)
        msg = (
            f"gazepoint collect  -  {self.mode} calibration, {len(self.targets)} targets, "
            f"{len(self.mons)} screen(s) ({n_ext} external)\n\n"
            "Look at each dot until it disappears.\n"
            "Keep your head natural; it's fine (good, even) to shift around between dots.\n\n"
            f"Face: {'OK' if self.face_ok else 'NOT DETECTED'}\n\n"
            "Space = start      Esc = abort"
        )
        c.create_text(m.w // 2, m.h // 2, text=msg, fill="white", font=("Sans", 16), justify="center")
        self.root.after(250, self._intro)

    def _show(self, i: int):
        if i >= len(self.targets):
            self.root.quit()
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

        def frame():
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
                if not self.face_ok and self.per_target.get(i, 0) == 0:
                    print(f"target {i}: no face frames")
                self.root.after(1, self._show, i + 1)
                return
            if not self.face_ok:
                c.create_text(m.w // 2, 30, text="face not detected", fill="orange", font=("Sans", 14))
            self.root.after(16, frame)

        frame()


def collect(cfg: dict, mode: str = "auto", rounds: int = 1, points: int | None = None, do_calibrate: bool = True):
    col = Collector(cfg, mode, rounds, points)
    print(f"session {col.writer.id}: mode={col.mode}, {len(col.targets)} targets")
    try:
        col.run()
    finally:
        col.stop = True
        time.sleep(0.2)
        col.writer.close()
        col.cam.close()
        col.tracker.close()
    got = sum(1 for v in col.per_target.values() if v)
    print(f"recorded {col.writer.count} frames, {got}/{len(col.targets)} targets with face frames")
    print(f"data: {col.writer.dir}")
    if col.aborted or not do_calibrate:
        return col.writer.id
    from .cli import run_calibrate

    run_calibrate(cfg, col.writer.id, col.mode, refit_user=False)
    return col.writer.id
