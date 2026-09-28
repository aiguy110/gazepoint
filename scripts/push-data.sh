#!/usr/bin/env bash
# Run ON the laptop: ship recorded sessions + calibration state to the dev box.
#   scripts/push-data.sh <rsync-dest>     e.g. bifrost:Projects/gazepoint/data/
#   scripts/push-data.sh                  no dest: write a tarball to move by hand
# Either way the dev box ends up with data/<laptop-hostname>/, usable via
#   GAZEPOINT_HOME=data/<laptop-hostname> uv run gazepoint sessions
set -euo pipefail
src="${GAZEPOINT_HOME:-$HOME/.local/share/gazepoint}"
host=$(hostname)
if [ $# -ge 1 ]; then
  rsync -az --info=stats1 --exclude models "$src/" "${1%/}/$host/"
  echo "pushed to ${1%/}/$host/"
else
  out="$HOME/gazepoint-data-$host-$(date +%Y%m%d-%H%M%S).tar.gz"
  tar -czf "$out" --exclude models -C "$(dirname "$src")" --transform "s|^$(basename "$src")|$host|" "$(basename "$src")"
  echo "wrote $out  (on the dev box: tar -xzf <it> -C data/)"
fi
