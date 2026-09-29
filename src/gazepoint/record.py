"""Session recording. A session is a directory under GAZEPOINT_HOME/sessions:

    session.json    metadata: kind, host, monitors, camera intrinsics, config snapshot
    samples.jsonl   one line per frame: {seq, t, frame, target, obs, ...}
    frames/*.jpg    raw webcam frames (kept so data can be reprocessed with new models)
    events.jsonl    usage sessions only: raw stream of {type, t, ...} records (see run.py)

`target` is {monitor, px: [x, y] desktop px, mm: [u, v] on that monitor, index}
and `obs` is a tracker.Observation dict (or null if no face / not yet processed).
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
from pathlib import Path

import cv2

from . import config
from .monitors import Monitor


class SessionWriter:
    """With async_frames, JPEG encoding happens on a worker thread so a live
    loop isn't slowed down; close() waits for the queue to drain."""

    def __init__(self, kind: str, meta: dict, jpeg_quality: int = 92, async_frames: bool = False):
        sid = time.strftime("%Y%m%d-%H%M%S") + f"-{kind}"
        self.id = sid
        self.dir = config.sessions_dir() / sid
        (self.dir / "frames").mkdir(parents=True)
        self.meta = {"id": sid, "kind": kind, "host": socket.gethostname(), "created": time.time(), **meta}
        (self.dir / "session.json").write_text(json.dumps(self.meta, indent=2, default=str))
        self._f = open(self.dir / "samples.jsonl", "a")
        self._ev = None
        self._lock = threading.Lock()
        self.q = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        self.count = 0
        self._saved: set[int] = set()
        self._queue: queue.Queue | None = None
        if async_frames:
            self._queue = queue.Queue()
            self._worker = threading.Thread(target=self._drain, daemon=True, name="frame-writer")
            self._worker.start()

    def _drain(self):
        while (item := self._queue.get()) is not None:
            cv2.imwrite(*item)

    def frame_path(self, seq: int) -> Path:
        return self.dir / "frames" / f"{seq:08d}.jpg"

    def save_frame(self, frame, seq: int) -> str:
        """Write frames/<seq>.jpg once (repeat calls are free) and return its name."""
        path = self.frame_path(seq)
        with self._lock:
            if seq in self._saved:
                return path.name
            self._saved.add(seq)
        item = (str(path), frame, self.q)
        if self._queue is not None:
            self._queue.put(item)
        else:
            cv2.imwrite(*item)
        return path.name

    def add(self, frame, seq: int, t: float, target: dict | None, obs: dict | None, **extra):
        name = self.save_frame(frame, seq)
        line = json.dumps({"seq": seq, "t": t, "frame": name, "target": target, "obs": obs, **extra})
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()
            self.count += 1

    def event(self, rec: dict):
        """Append one record to events.jsonl."""
        line = json.dumps(rec)
        with self._lock:
            if self._ev is None:
                self._ev = open(self.dir / "events.jsonl", "a")
            self._ev.write(line + "\n")
            self._ev.flush()

    def update_meta(self, **kv):
        self.meta.update(kv)
        (self.dir / "session.json").write_text(json.dumps(self.meta, indent=2, default=str))

    def close(self):
        if self._queue is not None:
            self._queue.put(None)
            self._worker.join()
        self._f.close()
        if self._ev:
            self._ev.close()


def resolve(session: str) -> Path:
    p = Path(session)
    if p.is_dir():
        return p
    p = config.sessions_dir() / session
    if p.is_dir():
        return p
    matches = sorted(config.sessions_dir().glob(f"*{session}*"))
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(f"session {session!r} not found (or ambiguous) under {config.sessions_dir()}")


def latest(kind: str | None = None) -> Path:
    ds = sorted(d for d in config.sessions_dir().iterdir() if d.is_dir() and (not kind or d.name.endswith(kind)))
    if not ds:
        raise SystemExit("no sessions recorded yet")
    return ds[-1]


def load(path: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((path / "session.json").read_text())
    recs = [json.loads(ln) for ln in open(path / "samples.jsonl") if ln.strip()]
    return meta, recs


def labelled(meta: dict, recs: list[dict]) -> list[dict]:
    """Frames with a target, minus presentations collect rejected as not fixated."""
    bad = set(meta.get("rejected_targets", []))
    return [r for r in recs if r.get("target") and r["target"].get("index") not in bad]


def monitors_of(meta: dict) -> list[Monitor]:
    return [Monitor(**m) for m in meta["monitors"]]
