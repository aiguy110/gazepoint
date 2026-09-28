#!/usr/bin/env bash
# Run a gazepoint command on the target inside its logged-in graphical session.
#   scripts/remote.sh <ssh-host> doctor
#   scripts/remote.sh <ssh-host> collect          (you still need to be at the laptop to look at the dots)
# Environment for the desktop session is reconstructed from /run/user/$UID, which
# covers GNOME (Wayland + XWayland) and plain X11 on :0.
set -euo pipefail
host=${1:?usage: remote.sh <ssh-host> <gazepoint args...>}
shift
dir=${GAZEPOINT_REMOTE_DIR:-gazepoint}
args=$(printf '%q ' "$@")
ssh -t "$host" "bash -lc '
  set -e
  export PATH=\$HOME/.local/bin:\$PATH
  rt=/run/user/\$(id -u)
  export XDG_RUNTIME_DIR=\$rt DBUS_SESSION_BUS_ADDRESS=unix:path=\$rt/bus
  export DISPLAY=\${DISPLAY:-:0}
  [ -S \$rt/wayland-0 ] && export WAYLAND_DISPLAY=wayland-0
  xa=\$(ls \$rt/.mutter-Xwaylandauth.* \$rt/gdm/Xauthority 2>/dev/null | head -1 || true)
  [ -n \"\$xa\" ] && export XAUTHORITY=\$xa
  cd $dir && exec uv run gazepoint $args
'"
