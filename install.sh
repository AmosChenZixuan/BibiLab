#!/usr/bin/env bash
# One-shot installer: build the (CPU-only) image and start Bibilab.
set -euo pipefail
cd "$(dirname "$0")"

# Preserve any developer-set keys (BIBILAB_PORT, HF_ENDPOINT, etc.); only the keys
# we manage here are written. Re-runs are idempotent.
#
# The strip list also purges keys this installer no longer writes. A box that ran the
# GPU-probing installer has COMPOSE_FILE=compose.yml:compose.cuda.yml in .env, and
# those overlay files are gone — leaving the key would make `docker compose up` fail
# to open a file that no longer exists. Dropping them lets compose fall back to
# compose.yml, so an upgrade from a GPU install lands on the CPU image.
existing_env=""
if [[ -f .env ]]; then
  existing_env=$(grep -v -E '^(TORCH_VARIANT|COMPOSE_FILE|UID|GID|RENDER_GID|VIDEO_GID)=' .env || true)
fi
{
  if [[ -n "$existing_env" ]]; then
    printf '%s\n' "$existing_env"
  fi
  printf 'UID=%s\n' "$(id -u)"
  printf 'GID=%s\n' "$(id -g)"
} > .env

# Create the bind-mount source before compose does. Docker auto-creates a missing
# host path as root:root, but the container runs as the host uid (compose `user:`)
# and would then fail to write the DB/caches under /data. A fresh one-click user
# has no ~/.bibilab yet, so this is the common case, not the edge.
mkdir -p "$HOME/.bibilab"

docker compose up --build -d
echo "Waiting for Bibilab to become healthy..."
for i in {1..30}; do
  if curl -fsS http://localhost:8765/health 2>/dev/null | grep -q '"overall"'; then
    echo "Bibilab is up at http://localhost:8765"
    exit 0
  fi
  sleep 1
done
echo "Bibilab did not become healthy in 30s. Check 'docker compose logs'." >&2
docker compose logs --tail=50 >&2
exit 1
