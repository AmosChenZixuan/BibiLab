#!/usr/bin/env bash
# One-shot installer: detect GPU once, build the matching image, start Bibilab.
set -euo pipefail
cd "$(dirname "$0")"

# Real passthrough probe: a working host nvidia-smi does NOT prove a container can
# see the GPU (on WSL2 you also need nvidia-container-toolkit wired into the daemon).
# Run nvidia-smi inside a throwaway --gpus container — exit 0 is the only proof.
if docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  TORCH_VARIANT=cuda
  COMPOSE_FILE=compose.yml:compose.cuda.yml
else
  TORCH_VARIANT=cpu
  COMPOSE_FILE=compose.yml
fi

cat > .env <<EOF
TORCH_VARIANT=$TORCH_VARIANT
COMPOSE_FILE=$COMPOSE_FILE
UID=$(id -u)
GID=$(id -g)
EOF

echo "GPU probe → $TORCH_VARIANT variant"
docker compose up --build -d
echo "Bibilab is starting at http://localhost:8765"
