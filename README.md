# gazepoint

Hold a hotkey and point with your head; tap a second key and the mouse jumps
to where you're looking, using only a laptop webcam. Head movements then
fine-tune the position. Linux only; built for a laptop that docks at several
desks with different external monitors.

How it works: MediaPipe finds the face and head pose, a gaze net (UniGaze-B or
ETH-XGaze) estimates where the eyes point, and a short calibration at each desk
maps that onto your monitors. On a jump, gazepoint warps the cursor to the median
gaze over a few frames. From then on the cursor follows your head, not your
noisy gaze. See `CLAUDE.md` for the design.

Requires Linux (X11, or GNOME Wayland), a webcam, and Python via
[uv](https://docs.astral.sh/uv/). An Intel NPU is optional; it makes the
larger gaze net fast.

## Quick start

```sh
git clone https://github.com/aiguy110/gazepoint ~/gazepoint
cd ~/gazepoint
scripts/bootstrap.sh                 # installs uv if needed, deps, models; reports permissions
sudo scripts/setup-input-perms.sh    # once: hotkey + virtual pointer access, then log out/in

uv run gazepoint doctor              # camera, monitors, permissions
uv run gazepoint preview             # check the arrows: look/turn right -> arrow/yaw go right (q quits)
uv run gazepoint collect             # dots on every screen, then auto-calibrate
uv run gazepoint run                 # hold the hotkey to point; Ctrl-C to quit
```

At a new desk, `collect` runs a full calibration; at a known one it runs a
quick one (about 5 dots per screen) to find where the laptop is this time.
`run` tells you which it needs. To let someone else use the laptop, add
`--profile <name>` to every command; each profile has its own gaze correction.

To update, run `git pull && scripts/bootstrap.sh`.

## Configuration

`src/gazepoint/config.toml` holds every setting with comments, and its values
are the defaults. They assume a keyboard that sends F14-F17 (the author's ZMK
Chocofi). Copy the keys you want to change into
`~/.config/gazepoint/config.toml`, e.g. for an ordinary keyboard:

```toml
[run]
hotkey = "KEY_RIGHTCTRL"
aim_mode = "jump"       # key down jumps to the gaze, then the head refines
```
