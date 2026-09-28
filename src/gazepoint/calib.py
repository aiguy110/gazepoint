"""Calibration: per-user gaze correction + screen geometry, and their persistence.

Three layers, persisted separately because they change on different timescales:

  user.json      A: 2x3 affine on the net's (yaw, pitch) in degrees. Absorbs the
                 user's kappa angle and model bias. Same at every desk.
  profiles/*     Constellation: pose of each external monitor relative to an
                 anchor monitor. One per set of external monitors (EDID ids).
  current.json   T_cam<-anchor for the current placement of the laptop.

The built-in panel is rigidly attached to the camera (both in the lid), so its
pose is fixed by laptop_panel_pose and it acts as the gauge that pins down A.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from . import config
from .gaze import angles_to_vec, vec_to_angles
from .geometry import (
    R_to_rvec,
    Screen,
    compose,
    hit_screens,
    invert,
    laptop_panel_pose,
    rvec_to_R,
)
from .monitors import Monitor, profile_hash, profile_key

A_IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


# ---------------------------------------------------------------- model


@dataclass
class Samples:
    """Flat arrays of labelled observations (one row per frame)."""

    screen: np.ndarray  # (n,) object array of screen ids
    uv: np.ndarray  # (n, 2) target on that screen, mm
    o: np.ndarray  # (n, 3) eye origin, camera mm
    raw: np.ndarray  # (n, 2) net (yaw, pitch) degrees
    fR: np.ndarray  # (n, 3, 3) model frame -> camera
    target_idx: np.ndarray  # (n,) which target presentation each frame belongs to

    @classmethod
    def from_records(cls, recs: list[dict]) -> "Samples":
        recs = [r for r in recs if r.get("obs")]
        return cls(
            screen=np.array([r["target"]["monitor"] for r in recs], dtype=object),
            uv=np.array([r["target"]["mm"] for r in recs], dtype=np.float64).reshape(-1, 2),
            o=np.array([r["obs"]["eye_origin"] for r in recs], dtype=np.float64).reshape(-1, 3),
            raw=np.array([r["obs"]["gaze_raw"] for r in recs], dtype=np.float64).reshape(-1, 2),
            fR=np.array([r["obs"]["frame_R"] for r in recs], dtype=np.float64).reshape(-1, 3, 3),
            target_idx=np.array([r["target"]["index"] for r in recs], dtype=np.int64),
        )

    def subset(self, mask) -> "Samples":
        return Samples(*(getattr(self, f)[mask] for f in self.__dataclass_fields__))

    def __len__(self):
        return len(self.screen)


def apply_A(A: np.ndarray, raw_deg: np.ndarray) -> np.ndarray:
    raw_deg = np.atleast_2d(raw_deg)
    return raw_deg @ A[:, :2].T + A[:, 2]


def gaze_dir(A: np.ndarray, raw_deg, fR) -> np.ndarray:
    """Corrected gaze direction(s) in camera coordinates."""
    ang = np.radians(apply_A(A, raw_deg))
    v = angles_to_vec(ang[:, 0], ang[:, 1])
    fR = np.asarray(fR).reshape(-1, 3, 3)
    return np.einsum("nij,nj->ni", fR, v)


def target_angles(P: np.ndarray, o: np.ndarray, fR: np.ndarray) -> np.ndarray:
    """(yaw, pitch) degrees, in the model frame, of the ray from o to P."""
    d = np.einsum("nji,nj->ni", fR, P - o)  # fR^T (P - o)
    yaw, pitch = vec_to_angles(d)
    return np.degrees(np.stack([yaw, pitch], axis=1))


def screens_from_monitors(mons: list[Monitor], cfg: dict) -> dict[str, Screen]:
    out = {}
    for m in mons:
        s = Screen(
            id=m.id,
            name=m.name,
            logical=(m.x, m.y, m.w, m.h),
            size_mm=(m.width_mm, m.height_mm),
            builtin=m.builtin,
        )
        if m.builtin:
            lc = cfg["laptop"]
            s.R, s.t = laptop_panel_pose(
                s.size_mm, lc["camera_above_panel_mm"], lc["camera_right_of_center_mm"]
            )
        out[m.id] = s
    return out


def builtin_id(mons: list[Monitor]) -> str:
    b = [m for m in mons if m.builtin]
    if len(b) != 1:
        raise RuntimeError(f"expected exactly one built-in panel, found {[m.connector for m in b]}")
    return b[0].id


# ---------------------------------------------------------------- fitting


def fit_A_linear(s: Samples, screens: dict[str, Screen], ridge: float = 20.0) -> np.ndarray:
    """Least squares for A on screens with known pose, shrinking the 2x2 part toward I.

    ridge is in 'equivalent samples': it's weak once there are hundreds of frames.
    Two passes: the second drops frames > 3 robust sigma off (blinks, saccades).
    """
    P = np.stack([screens[sid].point_cam(u, v) for sid, (u, v) in zip(s.screen, s.uv)])
    tgt = target_angles(P, s.o, s.fR)
    X = np.hstack([s.raw, np.ones((len(s), 1))])
    keep = np.ones(len(s), dtype=bool)
    A = A_IDENTITY.copy()
    for _ in range(2):
        Xk, Tk = X[keep], tgt[keep]
        reg = np.sqrt(ridge) * np.array([[1.0, 0, 0], [0, 1.0, 0]])
        for k in range(2):
            lhs = np.vstack([Xk, reg])
            rhs = np.concatenate([Tk[:, k], np.sqrt(ridge) * A_IDENTITY[k, :2]])
            A[k] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        err = np.linalg.norm(X @ A.T - tgt, axis=1)
        med = np.median(err)
        mad = np.median(np.abs(err - med)) * 1.4826 + 1e-6
        keep = err < med + 3 * mad
    return A


def init_pose_pnp(s: Samples, A: np.ndarray, size_mm) -> tuple[np.ndarray, np.ndarray]:
    """Initial screen pose from gaze bearings, treating the eye as a pinhole camera.

    Averages frames per target, rotates bearings into a frame looking along
    their mean, and solves planar PnP (IPPE). Falls back to 'facing the user
    650 mm away along the mean gaze' when PnP is degenerate or implausible.
    """
    d = gaze_dir(A, s.raw, s.fR)
    tids = np.unique(s.target_idx)
    obj, bear = [], []
    for t in tids:
        m = s.target_idx == t
        obj.append([*s.uv[m][0], 0.0])
        b = d[m].mean(axis=0)
        bear.append(b / np.linalg.norm(b))
    obj, bear = np.array(obj), np.array(bear)
    o = s.o.mean(axis=0)
    z = bear.mean(axis=0)
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    F = np.stack([x, np.cross(z, x), z], axis=1)  # x = user's right, y = down, z = along gaze

    def fallback():
        c = o + 650.0 * z
        return F.copy(), c - F @ np.array([size_mm[0] / 2, size_mm[1] / 2, 0.0])

    if len(obj) < 4:
        return fallback()
    b = bear @ F
    img = (b[:, :2] / b[:, 2:3]).astype(np.float64)
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, img, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE
        )
    except cv2.error:
        return fallback()
    if not n:
        return fallback()
    best = int(np.argmin(np.asarray(errs).ravel()))
    R = F @ rvec_to_R(rvecs[best])
    t = F @ tvecs[best].ravel() + o
    center = R @ np.array([size_mm[0] / 2, size_mm[1] / 2, 0.0]) + t
    dist = np.linalg.norm(center - o)
    if not (200.0 < dist < 2500.0) or R[:, 2] @ (center - o) <= 0:
        return fallback()
    return R, t


@dataclass
class Group:
    """Screens sharing one free 6-DoF pose: screen pose = G o offset[screen]."""

    R: np.ndarray
    t: np.ndarray
    offsets: dict[str, tuple[np.ndarray, np.ndarray]]


def joint_fit(
    s: Samples,
    screens: dict[str, Screen],
    groups: list[Group],
    A: np.ndarray,
    fit_A: bool,
    ridge: float = 20.0,
):
    """Nonlinear least squares over A (optional) and group poses; angular residuals.

    Updates screens' poses in place; returns (A, final cost).
    """
    fixed = [sid for sid in screens if not any(sid in g.offsets for g in groups)]
    n_A = 6 if fit_A else 0

    def unpack(p):
        A_ = p[:6].reshape(2, 3) if fit_A else A
        poses = {}
        for gi, g in enumerate(groups):
            q = p[n_A + 6 * gi : n_A + 6 * gi + 6]
            GR, Gt = rvec_to_R(q[:3]), q[3:]
            for sid, (oR, ot) in g.offsets.items():
                poses[sid] = compose(GR, Gt, oR, ot)
        for sid in fixed:
            poses[sid] = (screens[sid].R, screens[sid].t)
        return A_, poses

    sids = np.unique(s.screen)
    masks = {sid: s.screen == sid for sid in sids}
    X = np.hstack([s.raw, np.ones((len(s), 1))])

    def resid(p):
        A_, poses = unpack(p)
        pred = X @ A_.T
        tgt = np.empty_like(pred)
        for sid in sids:
            m = masks[sid]
            R, t = poses[sid]
            P = s.uv[m] @ R[:, :2].T + t
            tgt[m] = target_angles(P, s.o[m], s.fR[m])
        r = [(pred - tgt).ravel()]
        if fit_A:
            r.append(np.sqrt(ridge) * (A_[:, :2] - np.eye(2)).ravel())
        return np.concatenate(r)

    p0 = [A.ravel()] if fit_A else []
    p0 += [np.concatenate([R_to_rvec(g.R), g.t]) for g in groups]
    p0 = np.concatenate(p0) if p0 else np.zeros(0)
    if p0.size == 0:
        return A, 0.0
    res = least_squares(resid, p0, loss="soft_l1", f_scale=3.0, x_scale="jac")
    A_, poses = unpack(res.x)
    for gi, g in enumerate(groups):
        q = res.x[n_A + 6 * gi : n_A + 6 * gi + 6]
        g.R, g.t = rvec_to_R(q[:3]), q[3:]
    for sid, (R, t) in poses.items():
        screens[sid].R, screens[sid].t = R, t
    return A_, float(res.cost)


# ---------------------------------------------------------------- persistence


def _read(path):
    return json.loads(path.read_text()) if path.exists() else None


def _write(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.rename(path)


def _pose_json(R, t):
    return {"R": np.asarray(R).round(8).tolist(), "t": np.asarray(t).round(4).tolist()}


def _pose_load(d):
    return np.array(d["R"], dtype=np.float64), np.array(d["t"], dtype=np.float64)


def load_user() -> np.ndarray | None:
    d = _read(config.home() / "user.json")
    return np.array(d["A"]) if d else None


def load_profile(key: str) -> dict | None:
    return _read(config.profiles_dir() / f"{profile_hash(key)}.json")


def load_current() -> dict | None:
    return _read(config.home() / "current.json")


def pick_anchor(screens: dict[str, Screen]) -> str:
    ext = [s for s in screens.values() if not s.builtin]
    return max(ext, key=lambda s: (s.size_mm[0] * s.size_mm[1], s.id)).id


# ---------------------------------------------------------------- top level


class NotCalibrated(RuntimeError):
    pass


def calibrate(recs: list[dict], mons: list[Monitor], cfg: dict, mode: str, refit_user: bool, session: str):
    """Fit from a recorded session. mode: 'full' (new constellation) or 'quick'
    (re-place a known constellation). Returns (A, screens, report dict)."""
    s = Samples.from_records(recs)
    if len(s) == 0:
        raise RuntimeError("no usable frames (was a face detected?)")
    screens = screens_from_monitors(mons, cfg)
    s = s.subset(np.isin(s.screen, list(screens)))
    bid = builtin_id(mons)
    key = profile_key(mons)
    ext_ids = [sid for sid in screens if sid != bid]
    report: dict = {"mode": mode, "profile_key": key, "frames": len(s)}

    A_prev = load_user()
    lap = s.subset(s.screen == bid)
    fit_A = mode == "full" or refit_user or A_prev is None
    if fit_A and len(np.unique(lap.target_idx)) >= 4:
        A = fit_A_linear(lap, screens)
    else:
        A = A_prev if A_prev is not None else A_IDENTITY.copy()

    groups: list[Group] = []
    present = [sid for sid in ext_ids if np.any(s.screen == sid)]
    if mode == "full":
        missing = set(ext_ids) - set(present)
        if missing:
            raise RuntimeError(f"full calibration needs targets on every monitor; missing {missing}")
        for sid in ext_ids:
            R, t = init_pose_pnp(s.subset(s.screen == sid), A, screens[sid].size_mm)
            groups.append(Group(R, t, {sid: (np.eye(3), np.zeros(3))}))
        A, cost = joint_fit(s, screens, groups, A, fit_A=fit_A)
    elif ext_ids:
        prof = load_profile(key)
        if prof is None:
            raise NotCalibrated(f"no profile for this monitor set; run a full collect first ({key})")
        anchor = prof["anchor"]
        offsets = {sid: _pose_load(prof["monitors"][sid]) for sid in ext_ids}
        cands = []
        cur = load_current()
        if cur and cur.get("profile_key") == key:
            cands.append(_pose_load(cur))
        for sid in present:
            R, t = init_pose_pnp(s.subset(s.screen == sid), A, screens[sid].size_mm)
            cands.append(compose(R, t, *invert(*offsets[sid])))
        if not cands:
            raise RuntimeError("quick calibration needs targets on at least one external monitor")
        best = None
        for R0, t0 in cands:
            trial = {k: Screen(**{**v.__dict__}) for k, v in screens.items()}
            g = Group(R0, t0, offsets)
            A_fit, cost = joint_fit(s, trial, [g], A, fit_A=refit_user)
            if best is None or cost < best[1]:
                best = (A_fit, cost, trial, g)
        A, cost, screens, g = best
        groups = [g]
        report["anchor"] = anchor
    else:
        cost = 0.0

    report["cost"] = cost
    report["A"] = A.round(4).tolist()
    scales = np.linalg.svd(A[:, :2], compute_uv=False)
    if scales.min() < 0.5 or scales.max() > 2.0 or np.linalg.det(A[:, :2]) < 0:
        report["warning"] = (
            f"degenerate-looking gaze correction (singular values {scales.round(2).tolist()}); "
            "check `gazepoint preview` for sign/convention problems or a poorly lit face"
        )
    report["eval"] = evaluate(s, A, list(screens.values()))

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    if fit_A:
        _write(config.home() / "user.json", {"A": A.tolist(), "updated": now, "session": session})
    if ext_ids:
        anchor = pick_anchor(screens) if mode == "full" else report["anchor"]
        a = screens[anchor]
        if mode == "full":
            aRi, ati = invert(a.R, a.t)
            prof = {
                "key": key,
                "anchor": anchor,
                "monitors": {
                    sid: {
                        "name": screens[sid].name,
                        "size_mm": list(screens[sid].size_mm),
                        **_pose_json(*compose(aRi, ati, screens[sid].R, screens[sid].t)),
                    }
                    for sid in ext_ids
                },
                "updated": now,
                "session": session,
            }
            _write(config.profiles_dir() / f"{profile_hash(key)}.json", prof)
            report["anchor"] = anchor
        _write(
            config.home() / "current.json",
            {"profile_key": key, "anchor": anchor, **_pose_json(a.R, a.t), "updated": now, "session": session},
        )
    return A, screens, report


def load_runtime(mons: list[Monitor], cfg: dict) -> tuple[np.ndarray, list[Screen]]:
    """A and camera-frame screens for the currently connected monitors."""
    A = load_user()
    if A is None:
        raise NotCalibrated("no user calibration; run `gazepoint collect`")
    screens = screens_from_monitors(mons, cfg)
    key = profile_key(mons)
    ext_ids = [sid for sid, s in screens.items() if not s.builtin]
    if ext_ids:
        prof = load_profile(key)
        cur = load_current()
        if prof is None:
            raise NotCalibrated(f"unknown monitor set; run `gazepoint collect --mode full` ({key})")
        if cur is None or cur.get("profile_key") != key:
            raise NotCalibrated("laptop moved to this desk since last calibration; run `gazepoint collect --mode quick`")
        GR, Gt = _pose_load(cur)
        for sid in ext_ids:
            screens[sid].R, screens[sid].t = compose(GR, Gt, *_pose_load(prof["monitors"][sid]))
    return A, list(screens.values())


def evaluate(s: Samples, A: np.ndarray, screens: list[Screen]) -> dict:
    """Per-monitor error stats: angular (deg), on-screen (mm, px), wrong-screen rate."""
    by_id = {sc.id: sc for sc in screens}
    d = gaze_dir(A, s.raw, s.fR)
    out = {}
    for sid in np.unique(s.screen):
        m = np.flatnonzero(s.screen == sid)
        sc = by_id[sid]
        ang, mm, px, wrong = [], [], [], 0
        for i in m:
            P = sc.point_cam(*s.uv[i])
            want = (P - s.o[i]) / np.linalg.norm(P - s.o[i])
            ang.append(np.degrees(np.arccos(np.clip(want @ d[i], -1, 1))))
            hit = hit_screens(screens, s.o[i], d[i])
            if hit is None:
                wrong += 1
                continue
            hs, u, v, _ = hit
            if hs.id != sid:
                wrong += 1
                continue
            mm.append(np.hypot(u - s.uv[i][0], v - s.uv[i][1]))
            ppm = sc.logical[2] / sc.size_mm[0]
            px.append(mm[-1] * ppm)

        def stats(a):
            return {"median": round(float(np.median(a)), 2), "p90": round(float(np.percentile(a, 90)), 2)} if a else None

        out[sc.name + f" [{sid}]"] = {
            "frames": len(m),
            "deg": stats(ang),
            "mm": stats(mm),
            "px": stats(px),
            "wrong_screen": round(wrong / len(m), 3),
        }
    return out
