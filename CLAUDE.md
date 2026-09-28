# gazepoint

Hold a hotkey → the mouse jumps to where you're looking (laptop webcam gaze
estimation), then head movement fine-tunes the position while the key is held.
Linux only. Personal project; runs on a laptop that docks at several desks with
different external monitors.

## Machines

- **Target** = the laptop. Its built-in webcam is the only camera. This is the
  only place `collect`, `preview`, `run`, and `doctor` mean anything.
- **Dev box** = where code is edited (usually `bifrost`, RTX 3080, no webcam,
  no user desktop session). **bifrost cannot reach the laptop.** Code goes via
  GitHub (`aiguy110/gazepoint`): the laptop runs `git pull && scripts/bootstrap.sh`.
  Data comes back via `scripts/push-data.sh`, run on the laptop (rsync or a
  tarball), and lands in `data/<laptop-host>/`. Analyse it with
  `GAZEPOINT_HOME=data/<host> uv run gazepoint calibrate|eval|reprocess ...`.
- `scripts/deploy.sh`, `remote.sh`, and `pull-data.sh` are the ssh-push
  equivalents for when the dev box *can* reach the target. `remote.sh`
  reconstructs DISPLAY, DBUS, and XAUTHORITY from `/run/user/$UID`.

## Design (why it looks like this)

- **Latch, then refine.** When the key goes down, take the median gaze hit over
  `latch_ms`, warp there, and record the head pose. While the key is held,
  cursor = anchor + gain·Δhead. Gaze noise (≈2–4°) never touches the cursor
  after the jump. Head angles go through a One Euro filter. The cursor is not
  smoothed with a PID loop; that only adds lag.
- **Perception** (`tracker.py`): MediaPipe Face Landmarker gives landmarks,
  and `cv2.solvePnP` against MediaPipe's canonical face gives head pose and eye
  origin. The default gaze net is ETH-XGaze ResNet-18 (hysts/ptgaze). Its input
  is an ETH-XGaze *normalised* face (`gaze.normalize_face`): a virtual camera
  600 mm from the face centre, rolled to the head's x-axis, 224². It's
  published only as PyTorch weights, so `download-models` converts the
  safetensors to ONNX without torch (`resnet_onnx.py`). Its output is rotated
  into the same face-centred (yaw, pitch) the older Gaze360 nets
  (yakhyo/gaze-estimation, square crop) report, so calibration code is shared.
  In a 2026-09 comparison on recorded sessions, the Gaze360 nets failed in
  backlit scenes. ETH-XGaze held up, and UniGaze-B was better still but ~8×
  slower (the Phase 2 teacher). Observations hold only *uncalibrated* values
  plus `gaze_model`, so recorded data survives calibration changes and
  `reprocess` can swap models.
- **Three calibration layers** (`calib.py`), persisted separately because they
  change on different timescales:
  1. `user.json`: a 2×3 affine `A` on the net's (yaw, pitch) in degrees. It
     absorbs kappa and model bias and is the same at every desk. It's tagged
     with the gaze model it was fit for and ignored for any other model. Once
     it exists, a full calibration at a new desk reuses it and fits only
     monitor poses (`--refit-user` to refit). At some desks the laptop panel is
     a poor gauge, e.g. looked at steeply down through glare.
  2. `profiles/<sha1(key)[:12]>.json`: external monitor poses relative to an
     anchor monitor. The key is the sorted EDID ids of the external monitors
     (`monitors.profile_key`).
  3. `current.json`: T_cam←anchor for wherever the laptop was set down this time.
- **The built-in panel is the gauge.** The webcam and panel are both in the
  lid, so the panel pose is fixed (`geometry.laptop_panel_pose`). Laptop-panel
  targets pin down `A` without getting mixed up with desk geometry.
- **Full vs quick** (`collect --mode`): *full* (new user or unknown monitor
  set) fits `A` plus a 6-DoF pose per external monitor, initialised by PnP
  that treats the eye as a pinhole camera and gaze vectors as bearings, then
  refined jointly. *Quick* (known monitor set, laptop moved) fits only the
  6-DoF T_cam←anchor from about 5 targets per screen. `auto` picks one.
  After the last target, `collect` calibrates in place and switches to a live
  gaze view: a trail of raw hits, plus a ring where `run` would latch. **G**
  skips collection and uses the saved calibration.
- Fit quality is judged in gaze space, not by parameter recovery: `A`'s scale
  trades off against screen depth, and that trade-off is harmless.
- `calibrate` refuses to save a fit whose `A` has collapsed (the fitter can
  "explain" gaze that doesn't track the targets by making `A` ~0 and parking
  monitors along a constant direction) or that puts too many frames on the
  wrong screen (`[calibrate]` thresholds; `--force` overrides). A collapsed
  saved `user.json` is refit rather than reused.

## Conventions (get these right or everything silently breaks)

- Camera frame = OpenCV axes, **mm**: x = image right = *the user's left*,
  y = down, z = toward the user.
- Screen local frame: u = right as the user sees it, v = down, origin at the
  top-left of the active area. Pose maps local → camera.
- Gaze angles (`gaze.py`): d = [−cos p·sin y, −sin p, −cos p·cos y], so (0, 0)
  looks back at the camera. They are expressed in a *face-centred* virtual
  camera (`face_frame`) because Gaze360 crops are rectified that way.
- Head angles are user-centric: yaw > 0 = head turned to the user's right,
  pitch > 0 = head up.
- Desktop positions are logical px (xrandr / Mutter logical layout). Physical
  sizes come from EDID.
- Monitor id = `vendor:name-or-product:serial` from EDID (`monitors.edid_id`).

## Data layout (`GAZEPOINT_HOME`, default `~/.local/share/gazepoint`)

`sessions/<ts>-collect|usage/` holds `session.json` (monitors, K, config
snapshot), `samples.jsonl` (`{seq, t, frame, target, obs}`), and
`frames/*.jpg` (raw frames, kept so they can be reprocessed with new models).
Collect presentations that weren't a steady fixation are listed in
`session.json`'s `rejected_targets`. Their frames are kept, but `calibrate` and
`eval` skip them (`record.labelled`).
`usage` sessions come from `run`. When the key is released, the latch frames
are labelled with the final cursor position. Clicks logged with `--log-clicks`
(X11 only) are stored with `obs: null` until `gazepoint reprocess` fills them
in. Models live in `models/`. User config lives in
`~/.config/gazepoint/config.toml` (overrides `config.DEFAULTS`).

## Working on it

- **Ship by default.** Once a new feature or bug fix has been validated (tests
  pass and, where it matters, it has been run on the target), commit it,
  fast-forward merge it into `main`, and push to `origin` without asking first.
  The laptop picks up changes through `git pull`, so an unpushed fix doesn't
  exist as far as it's concerned.
- `uv run pytest`: synthetic end-to-end calibration (full, then quick after
  "moving the laptop"), EDID parsing, and the live-mode state machine. Keep
  these passing; add synthetic tests for new fitting logic.
- The collect UI can be driven headlessly: set `camera.index` to a video file
  path (it loops) and run under `xvfb-run -a -s "-screen 0 3200x1200x24"` with
  `collect.detect` monkeypatched to fake monitors.
- `pyproject.toml` forces uv-managed Python because the collect UI needs
  tkinter, which distro Pythons often lack.
- Never set a Tk `cursor=` (not even `"none"`). uv's Python aborts with an
  `xcb_xlib_unknown_seq_number` assertion, because its Tk has a static libX11
  that dlopens the system libXcursor. `collect.hide_cursor` hides the pointer
  over python-xlib's own connection instead.
- Use `evdev-binary` (prebuilt wheels), not `evdev`. The target may not have
  a compiler.
- Perception costs about 30 ms/frame on bifrost's CPU with ETH-XGaze
  (landmarks ~12 ms, gaze net ~17 ms at 224²; the old Gaze360 MobileOne net
  was ~10 ms at 448²). The laptop will be slower, so check `preview` fps.

## Not yet verified on real hardware (check these first)

- Gaze/head sign conventions against a live face (`preview` shows what to
  check). Calibration can absorb sign flips, but `A` shows up as degenerate
  (a warning in the calibrate report) if the data is garbage.
- Monitor detection: xrandr `--prop` EDID parsing on X11, and Mutter
  `GetCurrentState` over jeepney plus sysfs EDID matching on GNOME Wayland.
  Other Wayland compositors aren't supported.
- Fullscreen Tk windows landing on the right monitor. `collect` warns and
  records `window_geometry` if they don't. Fractional scaling on XWayland is
  likely wrong.
- The `uinput` absolute pointer on Wayland (it mimics QEMU's usb-tablet, and
  compositors should map it to the whole desktop). X11 uses XWarpPointer.
- Default hotkey is Right Ctrl (not grabbed, so apps still see it).
  `run` prints hotkey down, the jump, and hotkey up, plus why a jump is
  stalled. It also hints once if the opposite-side key is pressed instead.
  The listener rescans every 2 s for keyboards that (re)connect. The
  operator's ZMK keyboard uses home-row mods: Right Ctrl is a hold on the
  right pinky with a 200 ms tapping term, so expect that much lag before the
  hotkey registers.
  `scripts/setup-input-perms.sh` is needed for `/dev/input` and `/dev/uinput`.

## Roadmap

- **Phase 1:** refit `A`/T_cam←anchor online from `usage` sessions and click
  logs; re-anchor during a hold when the gaze makes a large saccade; camera
  intrinsics calibration (checkerboard) instead of `hfov_deg`.
- **Phase 2:** distill a larger teacher (UniGaze / ETH-XGaze) into a small
  student using public data (MPIIFaceGaze, EVE, ETH-XGaze, Gaze360) plus the
  recorded sessions; proper gaze data normalisation; a short temporal window.
