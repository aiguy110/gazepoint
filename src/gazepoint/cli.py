"""gazepoint command line."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

from . import calib, config, models, record
from .monitors import detect, desktop, profile_hash, profile_key, session_type


def _print_json(obj):
    print(json.dumps(obj, indent=2, default=str))


# ---------------------------------------------------------------- commands


def cmd_doctor(cfg, args):
    fails = 0

    def line(status, what, detail=""):
        nonlocal fails
        fails += status == "FAIL"
        print(f"[{status:4s}] {what}{': ' + detail if detail else ''}")

    line("ok", "python", f"{platform.python_version()} on {platform.platform()}")
    line("ok", "GAZEPOINT_HOME", str(config.home()))
    st = session_type()
    line("ok" if st != "unknown" else "FAIL", "session", f"{st} / {desktop() or '?'}")
    missing = models.check(cfg)
    line("FAIL" if missing else "ok", "models", "missing " + ", ".join(missing) if missing else "present")
    try:
        import onnxruntime as ort

        line("ok", "onnxruntime providers", ", ".join(ort.get_available_providers()))
    except Exception as e:  # noqa: BLE001
        line("FAIL", "onnxruntime", str(e))
    try:
        mons = detect()
        for m in mons:
            line("ok", "monitor", f"{m.name} [{m.id}] {m.connector} {m.w}x{m.h}+{m.x}+{m.y} "
                 f"{m.width_mm:.0f}x{m.height_mm:.0f}mm{' builtin' if m.builtin else ''}")
        n_builtin = sum(m.builtin for m in mons)
        line("ok" if n_builtin == 1 else "FAIL", "built-in panel", f"{n_builtin} found, exactly one required")
        key = profile_key(mons)
        prof = calib.load_profile(key)
        line("ok" if prof else "warn", "profile", f"{key} -> {'known' if prof else 'NEW (needs full collect)'}")
        try:
            calib.load_runtime(mons, cfg)
            line("ok", "calibration", "ready for this desk")
        except calib.NotCalibrated as e:
            line("warn", "calibration", str(e))
    except Exception as e:  # noqa: BLE001
        line("FAIL", "monitor detection", f"{type(e).__name__}: {e}")
    try:
        from .camera import Camera

        cam = Camera(cfg)
        seq, n, t0 = 0, 0, time.monotonic()
        while time.monotonic() - t0 < 2:
            _, _, seq = cam.next(seq)
            n += 1
        cam.close()
        line("ok", "camera", f"{cam.width}x{cam.height}, {n / 2:.1f} fps measured (driver says {cam.fps:.0f})")
    except Exception as e:  # noqa: BLE001
        line("FAIL", "camera", str(e))
    from .io import readable_input

    line("ok" if readable_input() else "warn", "read /dev/input", "" if readable_input() else "hotkey needs the input group")
    uinput_ok = os.access("/dev/uinput", os.W_OK)
    line("ok" if uinput_ok else ("warn" if st == "x11" else "FAIL"), "write /dev/uinput",
         "" if uinput_ok else "needed for the Wayland pointer backend")
    sys.exit(1 if fails else 0)


def cmd_monitors(cfg, args):
    mons = detect()
    key = profile_key(mons)
    _print_json({"session": session_type(), "profile_key": key, "profile_file": profile_hash(key) + ".json",
                 "monitors": [m.to_json() for m in mons]})


def run_calibrate(cfg, session: str, mode: str | None, refit_user: bool, force: bool = False):
    path = record.resolve(session)
    meta, recs = record.load(path)
    mons = record.monitors_of(meta)
    mode = mode or meta.get("mode", "full")
    recs = record.labelled(meta, recs)
    for r in recs:  # sessions recorded before observations named their model
        if r.get("obs") and not r["obs"].get("gaze_model"):
            r["obs"]["gaze_model"] = meta["config"]["gaze"]["model"]
    try:
        A, screens, report = calib.calibrate(recs, mons, cfg, mode, refit_user, meta["id"], force)
    except calib.UnreliableFit as e:
        _print_json(e.report)
        raise
    _print_json(report)
    return report


def cmd_calibrate(cfg, args):
    session = args.session or record.latest("collect").name
    try:
        run_calibrate(cfg, session, args.mode, args.refit_user, args.force)
    except calib.UnreliableFit as e:
        sys.exit(f"ERROR: {e}")


def cmd_eval(cfg, args):
    path = record.resolve(args.session or record.latest().name)
    meta, recs = record.load(path)
    mons = record.monitors_of(meta)
    A, screens = calib.load_runtime(mons, cfg)
    s = calib.Samples.from_records(record.labelled(meta, recs))
    print(f"evaluating {path.name} ({len(s)} frames) against current calibration")
    _print_json(calib.evaluate(s, A, screens))


def cmd_reprocess(cfg, args):
    """(Re)run perception on a session's saved frames, e.g. with a different gaze model."""
    import cv2

    from .tracker import Tracker

    path = record.resolve(args.session)
    meta, recs = record.load(path)
    K = np.array(meta["camera"]["K"])
    tr = Tracker(cfg, K, video=False)
    n = 0
    for r in recs:
        if r.get("obs") and not args.all:
            continue
        frame = cv2.imread(str(path / "frames" / r["frame"]))
        obs = tr.process(frame, r["t"], r["seq"], keep_landmarks=True) if frame is not None else None
        r["obs"] = obs.to_json() if obs else None
        n += 1
    tr.close()
    backup = path / f"samples.{int(time.time())}.jsonl.bak"
    (path / "samples.jsonl").rename(backup)
    with open(path / "samples.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"reprocessed {n} frames with gaze model {cfg['gaze']['model']}; previous file kept as {backup.name}")


def cmd_sessions(cfg, args):
    for d in sorted(config.sessions_dir().iterdir()):
        if not (d / "session.json").exists():
            continue
        meta = json.loads((d / "session.json").read_text())
        n = sum(1 for _ in open(d / "samples.jsonl"))
        print(f"{d.name}  {meta.get('host', '?'):12s} {meta.get('mode', ''):6s} {n:6d} frames  {meta.get('profile_key', '')}")


def cmd_collect(cfg, args):
    from .collect import collect

    collect(cfg, args.mode, args.rounds, args.points, do_calibrate=not args.no_calibrate)


def cmd_preview(cfg, args):
    from .preview import preview

    preview(cfg)


def cmd_run(cfg, args):
    from .run import Live

    if args.gain:
        cfg["run"]["gain_px_per_deg"] = args.gain
    try:
        Live(cfg, log_clicks=args.log_clicks or None).run()
    except calib.NotCalibrated as e:
        sys.exit(str(e))


def cmd_download(cfg, args):
    models.download(args.gaze or None)


def cmd_bench(cfg, args):
    """Gaze-net latency per model and thread count (a random input; no camera needed)."""
    import os
    import time

    import numpy as np

    from .gaze import GazeNet, NormalizedGazeNet

    names = args.models or [cfg["gaze"]["model"]]
    threads = [int(n) for n in args.threads.split(",")]
    print(f"{os.cpu_count()} logical CPUs; threads 0 = onnxruntime default")
    for name in names:
        if name not in models.GAZE_MODELS:
            sys.exit(f"unknown gaze model {name!r}; choose from {models.GAZE_MODELS}")
        models.fetch_gaze(name)
        path = str(models.gaze_model_path(name))
        for n in threads:
            net = NormalizedGazeNet(path, n) if models.is_normalized(name) else GazeNet(path, threads=n)
            face = np.random.randint(0, 255, (224, 224, 3), np.uint8)
            for _ in range(3):
                net(face)
            k = 20
            t = time.perf_counter()
            for _ in range(k):
                net(face)
            print(f"  {name:20s} threads {n:2d}: {(time.perf_counter() - t) / k * 1000:6.1f} ms", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(prog="gazepoint", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check environment, camera, monitors, permissions")
    d = sub.add_parser("download-models", help="fetch face landmarker + gaze net")
    d.add_argument("--gaze", nargs="*", help=f"gaze models to fetch ({', '.join(models.GAZE_MODELS)})")
    b = sub.add_parser("bench", help="time the gaze net(s) at several thread counts")
    b.add_argument("models", nargs="*", help="gaze models (default: the configured one)")
    b.add_argument("--threads", default="1,2,4,8,12,0", help="comma-separated thread counts")
    sub.add_parser("monitors", help="print detected monitors and profile key")
    sub.add_parser("preview", help="webcam preview with gaze/head overlays")
    c = sub.add_parser("collect", help="show fixation targets, record labelled frames, calibrate, then show live gaze")
    c.add_argument("--mode", choices=["auto", "full", "quick"], default="auto",
                   help="full = new desk/user (fits constellation), quick = re-place known desk")
    c.add_argument("--rounds", type=int, default=1)
    c.add_argument("--points", type=int, help="targets per monitor (default from config)")
    c.add_argument("--no-calibrate", action="store_true", help="just record (e.g. for held-out eval)")
    k = sub.add_parser("calibrate", help="fit calibration from a recorded session")
    k.add_argument("session", nargs="?", help="session id/path (default: latest collect)")
    k.add_argument("--mode", choices=["full", "quick"])
    k.add_argument("--refit-user", action="store_true", help="also refit the per-user correction in quick mode")
    k.add_argument("--force", action="store_true", help="save even if the fit looks unreliable")
    e = sub.add_parser("eval", help="error stats for a session against the current calibration")
    e.add_argument("session", nargs="?")
    r = sub.add_parser("reprocess", help="rerun perception on a session's saved frames")
    r.add_argument("session")
    r.add_argument("--all", action="store_true", help="also redo frames that already have observations")
    sub.add_parser("sessions", help="list recorded sessions")
    u = sub.add_parser("run", help="live: hold the hotkey to point")
    u.add_argument("--gain", type=float, help="px per degree of head rotation")
    u.add_argument("--log-clicks", action="store_true", help="X11: log frames at normal mouse clicks")
    args = p.parse_args(argv)
    cfg = config.load()
    {
        "doctor": cmd_doctor,
        "download-models": cmd_download,
        "bench": cmd_bench,
        "monitors": cmd_monitors,
        "preview": cmd_preview,
        "collect": cmd_collect,
        "calibrate": cmd_calibrate,
        "eval": cmd_eval,
        "reprocess": cmd_reprocess,
        "sessions": cmd_sessions,
        "run": cmd_run,
    }[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
