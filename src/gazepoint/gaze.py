"""Appearance-based gaze network + angle conventions.

Angle convention (Gaze360 as processed by GazeHub, which the pretrained nets
use), in OpenCV camera axes:

    d = [-cos(pitch) sin(yaw), -sin(pitch), -cos(pitch) cos(yaw)]

so (0, 0) looks straight back at the camera, yaw > 0 looks toward the user's
right, pitch > 0 looks up.
"""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def angles_to_vec(yaw: np.ndarray | float, pitch: np.ndarray | float) -> np.ndarray:
    """Radians -> unit vector(s). Broadcasts; output has a trailing axis of 3."""
    yaw, pitch = np.asarray(yaw), np.asarray(pitch)
    cp = np.cos(pitch)
    return np.stack([-cp * np.sin(yaw), -np.sin(pitch), -cp * np.cos(yaw)], axis=-1)


def vec_to_angles(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vector(s) -> (yaw, pitch) radians."""
    v = np.asarray(v, dtype=np.float64)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    pitch = np.arcsin(np.clip(-v[..., 1], -1, 1))
    yaw = np.arctan2(-v[..., 0], -v[..., 2])
    return yaw, pitch


def face_frame(center: np.ndarray) -> np.ndarray:
    """Rotation whose z-axis points from the camera at `center` (a virtual camera
    aimed at the face). Columns are the virtual camera's axes in camera coords."""
    z = center / np.linalg.norm(center)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


class GazeNet:
    """L2CS-style binned-regression net: face crop -> (yaw, pitch) radians."""

    def __init__(self, path: str, bins: int = 90, binwidth: float = 4.0, offset: float = 180.0):
        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.session = ort.InferenceSession(path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        self.size = (int(w) if isinstance(w, int) else 448, int(h) if isinstance(h, int) else 448)
        self.idx = np.arange(bins, dtype=np.float32)
        self.binwidth, self.offset = binwidth, offset
        self.providers = self.session.get_providers()

    def _decode(self, logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        return np.radians((p * self.idx).sum(axis=1) * self.binwidth - self.offset)

    def __call__(self, crop_bgr: np.ndarray) -> tuple[float, float]:
        img = cv2.cvtColor(cv2.resize(crop_bgr, self.size), cv2.COLOR_BGR2RGB)
        x = ((img.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
        yaw_logits, pitch_logits = self.session.run(None, {self.input_name: x})
        return float(self._decode(yaw_logits)[0]), float(self._decode(pitch_logits)[0])
