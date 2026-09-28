"""Synthetic end-to-end calibration: known user bias + screen poses -> recover them."""

import numpy as np
import pytest

from gazepoint import calib, config
from gazepoint.gaze import angles_to_vec, face_frame, vec_to_angles
from gazepoint.geometry import Screen, compose, laptop_panel_pose, rvec_to_R
from gazepoint.monitors import Monitor

CFG = config.DEFAULTS

LAPTOP = Monitor("BOE:laptop:", "laptop", "eDP-1", 0, 1440, 1920, 1200, 300.0, 190.0, True)
EXT = Monitor("DEL:U2720Q:ABC", "dell", "DP-1", 0, 0, 2560, 1440, 597.0, 336.0, False)
EXT2 = Monitor("LEN:T24:XYZ", "lenovo", "DP-2", 2560, 0, 1920, 1080, 527.0, 296.0, False)
A_TRUE = np.array([[1.08, 0.04, 2.5], [-0.03, 0.94, -3.0]])


def ext_poses(yaw_deg=0.0, shift=(0.0, 0.0, 0.0)):
    """External monitor poses in the camera frame for one laptop placement."""
    R0 = np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])
    tilt = rvec_to_R([np.radians(-8), 0, 0])  # tilted back a little
    R_dell, t_dell = R0 @ tilt, np.array([EXT.width_mm / 2 + 50, -60 - EXT.height_mm, -180.0])
    turn = rvec_to_R([0, np.radians(25), 0])  # side monitor angled toward the user
    R_len, t_len = R0 @ turn, np.array([-EXT.width_mm / 2 + 30, -60 - EXT2.height_mm, -170.0])
    # moving the laptop = moving the camera; apply the inverse to the world
    M = rvec_to_R([0, np.radians(yaw_deg), 0])
    s = np.asarray(shift)
    return {EXT.id: compose(M, s, R_dell, t_dell), EXT2.id: compose(M, s, R_len, t_len)}


def synth(poses, per_screen, frames=12, noise_deg=1.2, seed=0):
    rng = np.random.default_rng(seed)
    screens = calib.screens_from_monitors([LAPTOP, EXT, EXT2], CFG)
    for sid, (R, t) in poses.items():
        screens[sid].R, screens[sid].t = R, t
    Ainv = np.linalg.inv(np.vstack([A_TRUE, [0, 0, 1]]))[:2]
    recs, idx = [], 0
    for sid, sc in screens.items():
        for fx in np.linspace(0.08, 0.92, 3):
            for fy in np.linspace(0.08, 0.92, 3)[: per_screen // 3 + 1]:
                idx += 1
                u, v = fx * sc.size_mm[0], fy * sc.size_mm[1]
                P = sc.point_cam(u, v)
                for _ in range(frames):
                    o = np.array([0, -40, 520.0]) + rng.normal(0, [40, 25, 40])
                    fR = face_frame(o)
                    yaw, pitch = vec_to_angles(fR.T @ (P - o))
                    true = np.degrees([yaw, pitch])
                    raw = Ainv @ np.append(true, 1) + rng.normal(0, noise_deg, 2)
                    recs.append({
                        "target": {"monitor": sid, "mm": [u, v], "index": idx},
                        "obs": {"eye_origin": o.tolist(), "gaze_raw": raw.tolist(), "frame_R": fR.tolist()},
                    })
    return recs


@pytest.fixture(autouse=True)
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("GAZEPOINT_HOME", str(tmp_path))


def test_angle_roundtrip():
    y, p = np.radians(20), np.radians(-10)
    y2, p2 = vec_to_angles(angles_to_vec(y, p))
    assert np.allclose([y, p], [y2, p2])
    assert np.allclose(angles_to_vec(0, 0), [0, 0, -1])


def test_laptop_panel_pose():
    R, t = laptop_panel_pose((300, 190), 8, 0)
    s = Screen("x", "x", (0, 0, 1920, 1200), (300, 190), True, R, t)
    assert np.allclose(s.point_cam(150, 0), [0, 8, 0])  # top-center is right below the camera
    assert s.point_cam(300, 0)[0] < 0  # user's right = camera -x


def clean_error(poses, A, screens, seed):
    """Median angular error (deg) per screen of a fitted model on noise-free data."""
    s = calib.Samples.from_records(synth(poses, 9, noise_deg=0.0, seed=seed))
    return {k: v["deg"]["median"] for k, v in calib.evaluate(s, A, list(screens.values())).items()}


def test_full_then_quick():
    mons = [LAPTOP, EXT, EXT2]
    truth = ext_poses()
    A, screens, rep = calib.calibrate(synth(truth, 9), mons, CFG, "full", False, "s1")
    # Individual parameters trade off against each other (A's scale vs screen
    # depth), so judge the fit by what matters: gaze -> screen error vs truth.
    for name, err in clean_error(truth, A, screens, seed=5).items():
        assert err < 1.0, (name, err)
    for sid, (R, t) in truth.items():
        assert np.linalg.norm(R @ [150, 100, 0] + t - screens[sid].point_cam(150, 100)) < 60, sid
    assert all(st["wrong_screen"] < 0.1 for st in rep["eval"].values())

    # Put the laptop down somewhere else: ~10 deg turned, shifted 6 cm; few targets.
    moved = ext_poses(yaw_deg=10, shift=(60, 5, -30))
    A2, screens2, _ = calib.calibrate(synth(moved, 3, seed=1), mons, CFG, "quick", False, "s2")
    for name, err in clean_error(moved, A2, screens2, seed=6).items():
        assert err < 1.0, (name, err)

    # Runtime path reproduces the fitted geometry.
    _, rt = calib.load_runtime(mons, CFG)
    by = {s.id: s for s in rt}
    for sid in moved:
        assert np.allclose(by[sid].point_cam(10, 10), screens2[sid].point_cam(10, 10), atol=1e-3)


def test_fixation_problem():
    from gazepoint.collect import fixation_problem

    cc = CFG["collect"]
    rng = np.random.default_rng(0)
    steady = rng.normal([10.0, 20.0], 3.0, (15, 2))
    assert fixation_problem(steady, cc) is None
    late = np.vstack([steady[:10], rng.normal([-15.0, 20.0], 3.0, (5, 2))])  # saccaded late
    assert fixation_problem(late, cc) is not None
    assert fixation_problem(np.vstack([steady[:8], steady[8:] - [0, 25]]), cc) is not None
    assert "face frames" in fixation_problem(steady[:3], cc)


def test_occluded_monitor_not_placed_behind_lid():
    """The dell's bottom edge is 60 mm above the camera, just clear of the lid: the fit
    must not park it behind the built-in panel."""
    poses = ext_poses()
    recs = [r for r in synth(poses, per_screen=9) if r["target"]["monitor"] != EXT2.id]
    A, screens, rep = calib.calibrate(recs, [LAPTOP, EXT], CFG, "full", False, "t")
    lap, ext = screens[LAPTOP.id], screens[EXT.id]
    eye = np.array([0, -40, 520.0])
    P = np.array([ext.point_cam(u, v) for u in (0, EXT.width_mm) for v in (0, EXT.height_mm)])
    assert calib.occlusion_depth(lap, eye, P).max() < 5.0
    assert rep["eval"][f"dell [{EXT.id}]"]["wrong_screen"] < 0.05


def test_collapsed_fit_is_not_saved(tmp_path):
    """Raw gaze that ignores the targets (backlit face) must not overwrite calibration."""
    rng = np.random.default_rng(3)
    recs = synth(ext_poses(), 9)
    for r in recs:
        r["obs"]["gaze_raw"] = (np.array([20.0, -10.0]) + rng.normal(0, 4, 2)).tolist()
    with pytest.raises(calib.UnreliableFit) as e:
        calib.calibrate(recs, [LAPTOP, EXT, EXT2], CFG, "full", False, "bad")
    assert e.value.report["problems"]
    assert not (tmp_path / "user.json").exists()
    calib.calibrate(recs, [LAPTOP, EXT, EXT2], CFG, "full", False, "bad", force=True)
    assert (tmp_path / "user.json").exists()


def test_collapsed_saved_correction_is_refit():
    truth = ext_poses()
    mons = [LAPTOP, EXT, EXT2]
    calib.calibrate(synth(truth, 9), mons, CFG, "full", False, "s1")
    calib._write(config.home() / "user.json", {"A": [[0.08, 0, 0], [0, 0.01, 0]]})
    A, _, rep = calib.calibrate(synth(ext_poses(yaw_deg=6), 5, seed=1), mons, CFG, "quick", False, "s2")
    assert "refit_user" in rep
    assert np.abs(A[:, :2] - A_TRUE[:, :2]).max() < 0.15
