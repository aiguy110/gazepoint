#!/usr/bin/env bash
# Copy recorded sessions + calibration state from a target into ./data/<host>/.
# Afterwards, analyse locally with e.g.:
#   GAZEPOINT_HOME=data/<host> uv run gazepoint sessions
#   GAZEPOINT_HOME=data/<host> uv run gazepoint calibrate <session>
set -euo pipefail
host=${1:?usage: pull-data.sh <ssh-host>}
cd "$(dirname "$0")/.."
mkdir -p "data/$host"
rsync -az --info=stats1 --exclude models "$host:.local/share/gazepoint/" "data/$host/"
echo "pulled into data/$host"
