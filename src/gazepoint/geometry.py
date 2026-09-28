"""3D geometry. All camera-frame quantities use OpenCV axes and millimetres:

    x = image right (the *user's left*), y = image down, z = out of the lens toward the user.

A screen's local frame has u = right (as the user sees it), v = down, w = u x v
(into the screen, away from the user); the origin is the top-left corner of the
active area. A screen pose (R, t) maps local -> camera: p_cam = R @ [u, v, 0] + t.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def rvec_to_R(rvec) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]


def R_to_rvec(R) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, dtype=np.float64))[0].ravel()


def compose(Ra, ta, Rb, tb):
    """(Ra,ta) o (Rb,tb): apply b then a."""
    return Ra @ Rb, Ra @ tb + ta


def invert(R, t):
    return R.T, -R.T @ t


@dataclass
class Screen:
    id: str
    name: str
    logical: tuple[int, int, int, int]  # x, y, w, h in desktop (logical) px
    size_mm: tuple[float, float]  # physical w, h of the active area
    builtin: bool = False
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def px_to_mm(self, px: float, py: float) -> tuple[float, float]:
        x, y, w, h = self.logical
        return (px - x) / w * self.size_mm[0], (py - y) / h * self.size_mm[1]

    def mm_to_px(self, u: float, v: float) -> tuple[float, float]:
        x, y, w, h = self.logical
        return x + u / self.size_mm[0] * w, y + v / self.size_mm[1] * h

    def point_cam(self, u: float, v: float) -> np.ndarray:
        return self.R @ np.array([u, v, 0.0]) + self.t

    def intersect(self, o: np.ndarray, d: np.ndarray):
        """Ray o + s*d against the screen plane -> (u, v) in mm, or None if behind."""
        n = self.R[:, 2]
        denom = float(n @ d)
        if abs(denom) < 1e-9:
            return None
        s = float(n @ (self.t - o)) / denom
        if s <= 0:
            return None
        local = self.R.T @ (o + s * d - self.t)
        return float(local[0]), float(local[1])

    def outside_mm(self, u: float, v: float) -> float:
        """Distance (mm) from (u, v) to the active area; 0 if inside."""
        du = max(0.0, -u, u - self.size_mm[0])
        dv = max(0.0, -v, v - self.size_mm[1])
        return float(np.hypot(du, dv))


def laptop_panel_pose(size_mm, camera_above_mm: float, camera_right_mm: float):
    """Pose of the built-in panel in camera coordinates.

    The webcam sits in the lid bezel, so the panel is coplanar with the camera
    (z = 0) and directly below it. User's right is camera -x, down is camera +y.
    """
    R = np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])
    t = np.array([size_mm[0] / 2 + camera_right_mm, camera_above_mm, 0.0])
    return R, t


def hit_screens(screens: list[Screen], o: np.ndarray, d: np.ndarray):
    """Best screen hit for a gaze ray.

    Returns (screen, u_mm, v_mm, outside_mm) choosing a screen that contains the
    hit if any, else the one whose hit is closest to its edge. None if the ray
    hits no plane at all.
    """
    best = None
    for s in screens:
        uv = s.intersect(o, d)
        if uv is None:
            continue
        out = s.outside_mm(*uv)
        if best is None or out < best[3]:
            best = (s, uv[0], uv[1], out)
    return best


def clamp_to_screen(s: Screen, u: float, v: float) -> tuple[float, float]:
    return min(max(u, 0.0), s.size_mm[0]), min(max(v, 0.0), s.size_mm[1])
