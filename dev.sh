#!/bin/bash
# Start both backend and web dev servers with prefixed logs

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "Starting backend (uv run uvicorn bibilab.main --reload)..."
echo "Starting web (npm run dev)..."
echo ""
echo "Press Ctrl+C to stop both."
echo "================================"

# Kill both on Ctrl+C
trap 'kill 0' INT

(
  cd "$ROOT/backend"
  PYTHONUNBUFFERED=1 uv run uvicorn bibilab.main:app --host 0.0.0.0 --port 8765 --reload 2>&1
) | sed -u 's/^/[backend] /' &

(
  cd "$ROOT/web"
  npm run dev 2>&1
) | sed 's/^/[web] /' &

wait
