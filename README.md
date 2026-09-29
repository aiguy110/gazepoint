# gazepoint

Hold Right Ctrl and the cursor jumps to where you're looking (laptop webcam).
Keep holding and move your head to fine-tune the position. Linux, Phase 0
prototype. See CLAUDE.md for the design.

## Quick start (on the laptop)

```sh
git clone https://github.com/aiguy110/gazepoint ~/gazepoint
cd ~/gazepoint
scripts/bootstrap.sh                 # installs uv if needed, deps, models; reports permissions
sudo scripts/setup-input-perms.sh    # once: hotkey + virtual pointer access, then log out/in

uv run gazepoint doctor              # camera, monitors, permissions
uv run gazepoint preview             # check the arrows: look/turn right -> arrow/yaw go right (q quits)
uv run gazepoint collect             # dots on every screen, then auto-calibrate
uv run gazepoint run                 # hold Right Ctrl to point; Ctrl-C to quit
```

To update later, run `git pull && scripts/bootstrap.sh`.

`bootstrap.sh` does the same-host version of `deploy.sh`. It is idempotent
and safe to re-run.

## Day to day

- At a new desk (a set of monitors it hasn't seen), `collect` runs a full
  calibration.
- When you return to a known desk, `collect` runs a quick one (about 5 dots per
  screen) to find where the laptop is this time.
- `run` refuses to start if the laptop hasn't been placed at this desk yet, and
  tells you which of the two to run.

To measure real error, record a held-out session and evaluate against it:

```sh
uv run gazepoint collect --no-calibrate
uv run gazepoint eval
```

Settings (hotkey, gain, camera, etc.) go in `~/.config/gazepoint/config.toml`.
Any key from `DEFAULTS` in `src/gazepoint/config.py` can be overridden, e.g.:

```toml
[run]
hotkey = "KEY_RIGHTALT"
gain_px_per_deg = 50
```

With the Chocofi's gazepoint firmware (`aiguy110/miryoku_zmk`: holding the
MOUSE thumb also holds F14, and `'` on that layer is F15):

```toml
[run]
hotkey = "KEY_F14"
reaim_key = "KEY_F15"   # jump to the gaze again mid-hold, e.g. to finish a drag
```

## Getting data off the laptop

Recorded sessions live in `~/.local/share/gazepoint/sessions`.

```sh
scripts/push-data.sh bifrost:Projects/gazepoint/data/   # rsync, if the laptop can reach the dev box
scripts/push-data.sh                                    # otherwise: writes ~/gazepoint-data-<host>-<date>.tar.gz
```

On the dev box, place it under `data/<laptop-host>/` (the tarball does this
itself: `tar -xzf <file> -C data/`), then:

```sh
GAZEPOINT_HOME=data/<laptop-host> uv run gazepoint sessions
GAZEPOINT_HOME=data/<laptop-host> uv run gazepoint calibrate <session>
```

## Deploying from another machine over ssh

If the dev box can reach the laptop, use these instead:

- `scripts/deploy.sh <host>` rsyncs the code and runs `bootstrap.sh` there.
- `scripts/remote.sh <host> <cmd>` runs a gazepoint command inside the laptop's
  desktop session.
- `scripts/pull-data.sh <host>` copies the data back.
