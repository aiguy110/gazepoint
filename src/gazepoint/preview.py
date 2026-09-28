"""Webcam preview with perception overlays, for sanity-checking conventions.

Checks worth doing the first time on a new machine:
  * Turn your head to YOUR right: 'head yaw' should go positive.
  * Look to your right without turning: the gaze arrow should point to your
    right in the (mirrored) preview, and 'gaze yaw' should go positive.
  * Distance should read roughly your eye-to-camera distance in mm.
If signs are wrong, the model/axis conventions in gaze.py need fixing.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from . import calib
from .calib import gaze_dir
from .camera import Camera
from .monitors import detect
from .run import gaze_px
from .tracker import Tracker


def _project(K, p):
    p = np.asarray(p, dtype=np.float64)
    return int(K[0, 0] * p[0] / p[2] + K[0, 2]), int(K[1, 1] * p[1] / p[2] + K[1, 2])


def preview(cfg: dict):
    cam = Camera(cfg)
    tr = Tracker(cfg, cam.K)
    try:
        A, screens = calib.load_runtime(detect(), cfg)
        status = "calibrated"
    except Exception as e:  # noqa: BLE001 - preview works uncalibrated too
        A, screens, status = calib.A_IDENTITY, None, f"uncalibrated ({e})"
    print(status)
    seq, n, t0, fps = 0, 0, time.monotonic(), 0.0
    while True:
        frame, t, seq = cam.next(seq)
        tp = time.monotonic()
        obs = tr.process(frame, t, seq)
        dt_ms = (time.monotonic() - tp) * 1000
        n += 1
        if time.monotonic() - t0 > 1:
            fps, n, t0 = n / (time.monotonic() - t0), 0, time.monotonic()
        vis = frame.copy()
        lines = [f"{fps:.1f} fps  proc {dt_ms:.0f} ms  {status}"]
        if obs:
            o = np.array(obs.eye_origin)
            x0, y0, x1, y1 = obs.bbox
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 1)
            for A_, color in ((calib.A_IDENTITY, (0, 0, 255)), (A, (255, 200, 0))):
                d = gaze_dir(A_, np.array(obs.gaze_raw), np.array(obs.frame_R))[0]
                cv2.arrowedLine(vis, _project(cam.K, o), _project(cam.K, o + 120 * d), color, 2)
            R = np.array(obs.head_R)
            hc = np.array(obs.head_t)
            for axis, color in ((0, (0, 0, 255)), (1, (0, 255, 0)), (2, (255, 0, 0))):
                cv2.line(vis, _project(cam.K, hc), _project(cam.K, hc + 60 * R[:, axis]), color, 1)
            lines += [
                f"head yaw {obs.head_yaw:+.1f}  pitch {obs.head_pitch:+.1f}",
                f"gaze yaw {obs.gaze_raw[0]:+.1f}  pitch {obs.gaze_raw[1]:+.1f} (raw)",
                f"distance {np.linalg.norm(o):.0f} mm",
            ]
            if screens:
                hit = gaze_px(A, screens, obs)
                if hit:
                    lines.append(f"-> {hit[0].name} ({hit[1]:.0f}, {hit[2]:.0f}) px")
        else:
            lines.append("no face")
        vis = cv2.flip(vis, 1)  # mirror for display only; drawn geometry is unmirrored
        for i, s in enumerate(lines):
            cv2.putText(vis, s, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, "red = raw gaze, orange = calibrated, q = quit", (10, vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("gazepoint preview", vis)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cam.close()
    tr.close()
    cv2.destroyAllWindows()
