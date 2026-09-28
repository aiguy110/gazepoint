"""Per-frame perception: MediaPipe landmarks -> head pose (PnP) -> face crop -> gaze net.

Produces an Observation holding only *uncalibrated* quantities; the per-user
correction and screen geometry are applied downstream (calib.py), so recorded
observations stay valid when calibration changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from . import models
from .gaze import GazeNet, face_frame

# Canonical-model vertices used to place the eyes (outer/inner corners, lids).
_RIGHT_EYE = [33, 133, 159, 145]
_LEFT_EYE = [362, 263, 386, 374]


def intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)


def load_canonical(path) -> np.ndarray:
    """468x3 canonical face in mm, converted to OpenCV-style axes (y down, z into face)."""
    verts = [list(map(float, ln.split()[1:4])) for ln in open(path) if ln.startswith("v ")]
    v = np.array(verts[:468]) * 10.0  # cm -> mm
    return v * np.array([1.0, -1.0, -1.0])


def head_angles(R: np.ndarray) -> tuple[float, float]:
    """(yaw, pitch) degrees of the face's forward direction, user-centric:
    yaw > 0 = head turned to the user's right, pitch > 0 = head tilted up."""
    f = R @ np.array([0.0, 0.0, -1.0])
    yaw = np.degrees(np.arctan2(-f[0], -f[2]))
    pitch = np.degrees(np.arctan2(-f[1], np.hypot(f[0], f[2])))
    return float(yaw), float(pitch)


@dataclass
class Observation:
    t: float  # capture time (time.monotonic seconds)
    seq: int
    head_R: list  # 3x3, face -> camera
    head_t: list  # mm
    head_yaw: float  # degrees, user-centric
    head_pitch: float
    eye_origin: list  # mm, camera frame (midpoint of the eyes)
    bbox: list  # face crop x0, y0, x1, y1 (px)
    gaze_raw: list  # [yaw, pitch] degrees, model output in the model's frame
    frame_R: list  # 3x3, model frame -> camera (identity if face_centered is off)
    landmarks: list | None = None  # 478x2 px, only kept when recording

    def to_json(self) -> dict:
        return asdict(self)


class Tracker:
    def __init__(self, cfg: dict, K: np.ndarray, video: bool = True):
        self.cfg = cfg
        self.K = K
        self.video = video
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(models.face_landmarker_path())),
            running_mode=mode,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.canonical = load_canonical(models.canonical_face_path())
        self.eye_mid = (
            self.canonical[_RIGHT_EYE].mean(axis=0) + self.canonical[_LEFT_EYE].mean(axis=0)
        ) / 2
        self.net = GazeNet(str(models.gaze_model_path(cfg["gaze"]["model"])))
        self._rvec = None
        self._tvec = None
        self._last_ms = -1

    def close(self):
        self.landmarker.close()

    def _landmarks(self, frame: np.ndarray, t: float):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.video:
            ms = max(int(t * 1000), self._last_ms + 1)
            self._last_ms = ms
            res = self.landmarker.detect_for_video(img, ms)
        else:
            res = self.landmarker.detect(img)
        if not res.face_landmarks:
            return None
        h, w = frame.shape[:2]
        return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float64)

    def _pose(self, lm: np.ndarray):
        guess = self._rvec is not None and self.video
        ok, rvec, tvec = cv2.solvePnP(
            self.canonical,
            lm[:468],
            self.K,
            None,
            rvec=self._rvec.copy() if guess else None,
            tvec=self._tvec.copy() if guess else None,
            useExtrinsicGuess=guess,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or tvec[2, 0] <= 0:
            self._rvec = self._tvec = None
            return None
        self._rvec, self._tvec = rvec, tvec
        return cv2.Rodrigues(rvec)[0], tvec.ravel()

    def crop(self, frame: np.ndarray, lm: np.ndarray):
        x0, y0 = lm.min(axis=0)
        x1, y1 = lm.max(axis=0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) * self.cfg["gaze"]["crop_scale"] / 2
        h, w = frame.shape[:2]
        bx0, by0 = int(max(0, cx - half)), int(max(0, cy - half))
        bx1, by1 = int(min(w, cx + half)), int(min(h, cy + half))
        return frame[by0:by1, bx0:bx1], [bx0, by0, bx1, by1]

    def process(self, frame: np.ndarray, t: float, seq: int, keep_landmarks: bool = False):
        lm = self._landmarks(frame, t)
        if lm is None:
            return None
        pose = self._pose(lm)
        if pose is None:
            return None
        R, tv = pose
        eye_origin = R @ self.eye_mid + tv
        crop, bbox = self.crop(frame, lm)
        if crop.size == 0:
            return None
        yaw, pitch = self.net(crop)
        fR = face_frame(eye_origin) if self.cfg["gaze"]["face_centered"] else np.eye(3)
        hy, hp = head_angles(R)
        return Observation(
            t=t,
            seq=seq,
            head_R=R.round(6).tolist(),
            head_t=tv.round(3).tolist(),
            head_yaw=round(hy, 4),
            head_pitch=round(hp, 4),
            eye_origin=eye_origin.round(3).tolist(),
            bbox=bbox,
            gaze_raw=[round(float(np.degrees(yaw)), 4), round(float(np.degrees(pitch)), 4)],
            frame_R=fR.round(6).tolist(),
            landmarks=lm.round(2).tolist() if keep_landmarks else None,
        )
