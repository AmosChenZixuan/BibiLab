#!/usr/bin/env bash
# One-shot installer: detect GPU once, build the matching image, start Bibilab.
set -euo pipefail
cd "$(dirname "$0")"

# Real passthrough probe: a working host nvidia-smi does NOT prove a container can
# see the GPU. On native Linux Docker Engine you need nvidia-container-toolkit wired
# into the daemon (Docker Desktop's WSL2 backend bundles it). Run nvidia-smi inside
# a throwaway --gpus container — exit 0 is the strongest probe we can run without
# baking probe logic into the image. Soft-fails are still possible; if a later
# error says "no CUDA device" anyway, the cpu image just runs slower and /health
# surfaces the cuda state.
if probe_output=$(docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi 2>&1); then
  TORCH_VARIANT=cuda
  COMPOSE_FILE=compose.yml:compose.cuda.yml
elif command -v rocminfo >/dev/null && rocminfo >/dev/null 2>&1; then
  # AMD/ROCm. Unlike NVIDIA there is no container-toolkit shim that can be missing —
  # container GPU access is plain device passthrough (/dev/kfd + /dev/dri, granted in
  # compose.rocm.yml). So a working host rocminfo is sufficient; no throwaway-container
  # probe needed. ROCm torch drives the AMD GPU through the same `cuda` device API.
  TORCH_VARIANT=rocm
  COMPOSE_FILE=compose.yml:compose.rocm.yml
  # /dev/kfd + /dev/dri are owned by the host's render/video groups; the container
  # must join those groups by *numeric* GID. Group names won't work: the slim runtime
  # image has no `render` group (container fails to start), and even a matching name
  # would resolve to a container-side GID that need not equal the host device owner.
  # Pass the host GIDs to compose.rocm.yml via .env.
  RENDER_GID=$(getent group render | cut -d: -f3 || true)
  VIDEO_GID=$(getent group video | cut -d: -f3 || true)
else
  TORCH_VARIANT=cpu
  COMPOSE_FILE=compose.yml
  echo "GPU probe failed; falling back to cpu variant." >&2
  echo "$probe_output" | tail -5 >&2
  if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    echo "Host has an NVIDIA GPU but the container can't see it — likely missing nvidia-container-toolkit." >&2
  fi
fi

# Preserve any developer-set keys (BIBILAB_PORT, HF_ENDPOINT, etc.); only the keys
# we manage here are written. Re-runs are idempotent.
existing_env=""
if [[ -f .env ]]; then
  existing_env=$(grep -v -E '^(TORCH_VARIANT|COMPOSE_FILE|UID|GID|RENDER_GID|VIDEO_GID)=' .env || true)
fi
{
  if [[ -n "$existing_env" ]]; then
    printf '%s\n' "$existing_env"
  fi
  printf 'TORCH_VARIANT=%s\n' "$TORCH_VARIANT"
  printf 'COMPOSE_FILE=%s\n' "$COMPOSE_FILE"
  printf 'UID=%s\n' "$(id -u)"
  printf 'GID=%s\n' "$(id -g)"
  if [[ "$TORCH_VARIANT" == rocm ]]; then
    printf 'RENDER_GID=%s\n' "$RENDER_GID"
    printf 'VIDEO_GID=%s\n' "$VIDEO_GID"
  fi
} > .env

echo "GPU probe → $TORCH_VARIANT variant"

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
