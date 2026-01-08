#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "✅ Activating venv"
source .venv/bin/activate

echo "✅ Starting Postgres (docker compose)"
docker compose up -d

echo "✅ Freeing port 8000 (if in use)"
PIDS=$(lsof -ti tcp:8000 || true)
if [ -n "${PIDS}" ]; then
  echo "⚠️ Port 8000 in use by PID(s): ${PIDS}. Killing..."
  kill -9 ${PIDS} || true
fi

echo "✅ Batch scoring (uses saved artifacts + marts)"
python -u ml/batch_score.py

echo "✅ Starting API at http://127.0.0.1:8000/docs"
exec uvicorn api.main:app --reload --port 8000
