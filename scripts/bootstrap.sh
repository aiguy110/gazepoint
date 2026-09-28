#!/usr/bin/env bash
# Run ON the target laptop (deploy.sh does this over ssh). Idempotent.
# Installs uv if missing, syncs the venv, downloads models, reports permissions.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null; then
  echo "== installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "== uv sync"
uv sync --quiet

echo "== models"
uv run --quiet gazepoint download-models

echo "== permissions"
id -nG | tr ' ' '\n' | grep -qx input \
  && echo "in input group: ok" \
  || echo "NOT in input group -> run: sudo scripts/setup-input-perms.sh (then log out/in)"
[ -w /dev/uinput ] && echo "/dev/uinput writable: ok" || echo "/dev/uinput not writable (only needed on Wayland) -> sudo scripts/setup-input-perms.sh"
echo "== done. Next: uv run gazepoint doctor"
