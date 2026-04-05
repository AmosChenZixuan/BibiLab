#!/bin/bash
# Start both backend and web dev servers with prefixed logs

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "Starting backend (uv run python -m bibilab.main)..."
echo "Starting web (npm run dev)..."
echo ""
echo "Press Ctrl+C to stop both."
echo "================================"

# Kill both on Ctrl+C
trap 'kill 0' INT

(
  cd "$ROOT/backend"
  PYTHONUNBUFFERED=1 uv run python -m bibilab.main 2>&1
) | sed -u 's/^/[backend] /' &

(
  cd "$ROOT/web"
  npm run dev 2>&1
) | sed 's/^/[web] /' &

wait
