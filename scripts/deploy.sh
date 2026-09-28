#!/usr/bin/env bash
# Push this checkout to a target machine and bootstrap it.
#   scripts/deploy.sh <ssh-host> [remote-dir]      (default remote-dir: gazepoint)
set -euo pipefail
host=${1:?usage: deploy.sh <ssh-host> [remote-dir]}
dir=${2:-gazepoint}
cd "$(dirname "$0")/.."
rsync -az --delete \
  --exclude .git --exclude .venv --exclude data --exclude __pycache__ --exclude .pytest_cache \
  ./ "$host:$dir/"
ssh "$host" "cd $dir && ./scripts/bootstrap.sh"
