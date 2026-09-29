"""Per-frame perception: MediaPipe landmarks -> head pose (PnP) -> face image -> gaze net.

`head` and `gaze` split the two halves so the gaze net can run on its own thread.

Produces an Observation holding only *uncalibrated* quantities; the per-user
correction and screen geometry are applied downstream (calib.py), so recorded
observations stay valid when calibration changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from . import models
from .gaze import GazeNet, NormalizedGazeNet, face_frame, normalize_face, vec_to_angles

# Canonical-model vertices used to place the eyes (outer/inner corners, lids).
_RIGHT_EYE = [33, 133, 159, 145]
_LEFT_EYE = [362, 263, 386, 374]
# ETH-XGaze's face centre: mean of the eye-corner centre and the nostril centre.
_EYE_CORNERS = [33, 133, 362, 263]
_NOSTRILS = [98, 327]


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
    gaze_raw: list | None  # [yaw, pitch] degrees, model output in the model's frame
    frame_R: list | None  # 3x3, model frame -> camera (identity if face_centered is off)
    landmarks: list | None = None  # 478x2 px, only kept when recording
    gaze_model: str | None = None  # None in sessions recorded before this was stored

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Pending:
    """A head-only Observation plus what the gaze net needs (Tracker.head -> Tracker.gaze)."""

    obs: Observation
    face: np.ndarray  # normalised face, or square crop for the Gaze360 nets
    Rn: np.ndarray | None  # camera -> normalised camera (normalised models only)
    eye_origin: np.ndarray


class Tracker:
    def __init__(self, cfg: dict, K: np.ndarray, video: bool = True):
        self.cfg = cfg
        self.K = K
        self.video = video
        models.ensure(cfg)
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
        self.model = cfg["gaze"]["model"]
        self.normalized = models.is_normalized(self.model)
        path = str(models.gaze_model_path(self.model))
        threads, device = cfg["gaze"]["threads"], cfg["gaze"]["device"]
        if self.normalized:
            self.net = NormalizedGazeNet(path, threads, device)
        else:
            self.net = GazeNet(path, threads=threads, device=device)
        c = self.canonical
        self.face_center = (c[_EYE_CORNERS].mean(axis=0) + c[_NOSTRILS].mean(axis=0)) / 2
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
        p = self.head(frame, t, seq, keep_landmarks)
        return self.gaze(p) if p is not None else None

    def head(self, frame: np.ndarray, t: float, seq: int, keep_landmarks: bool = False) -> Pending | None:
        """Landmarks and head pose (~12 ms), plus the gaze net's input. The
        Observation's gaze fields stay None until `gaze` fills them in."""
        lm = self._landmarks(frame, t)
        if lm is None:
            return None
        pose = self._pose(lm)
        if pose is None:
            return None
        R, tv = pose
        eye_origin = R @ self.eye_mid + tv
        if self.normalized:
            face, Rn = normalize_face(frame, self.K, R[:, 0], R @ self.face_center + tv)
            x0, y0 = lm.min(axis=0)
            x1, y1 = lm.max(axis=0)
            bbox = [int(x0), int(y0), int(x1), int(y1)]
        else:
            face, bbox = self.crop(frame, lm)
            Rn = None
            if face.size == 0:
                return None
        hy, hp = head_angles(R)
        obs = Observation(
            t=t,
            seq=seq,
            head_R=R.round(6).tolist(),
            head_t=tv.round(3).tolist(),
            head_yaw=round(hy, 4),
            head_pitch=round(hp, 4),
            eye_origin=eye_origin.round(3).tolist(),
            bbox=bbox,
            gaze_raw=None,
            frame_R=None,
            landmarks=lm.round(2).tolist() if keep_landmarks else None,
            gaze_model=None,
        )
        return Pending(obs, face, Rn, eye_origin)

    def gaze(self, p: Pending) -> Observation:
        """Run the gaze net on a `head` result. Thread-safe against `head`, so
        it can run on a worker thread."""
        if self.normalized:
            # Gaze in the normalised camera -> camera -> the same face-centred
            # angles the Gaze360 path reports, so calibration code is shared.
            fR = face_frame(p.eye_origin)
            yaw, pitch = vec_to_angles(fR.T @ (p.Rn.T @ self.net(p.face)))
        else:
            yaw, pitch = self.net(p.face)
            fR = face_frame(p.eye_origin) if self.cfg["gaze"]["face_centered"] else np.eye(3)
        return replace(
            p.obs,
            gaze_raw=[round(float(np.degrees(yaw)), 4), round(float(np.degrees(pitch)), 4)],
            frame_R=fR.round(6).tolist(),
            gaze_model=self.model,
        )
