#!/usr/bin/env bash
# Dev launcher: API on :8000, frontend on :5173 (vite dev server)
#
# 注记（v3.2）：后端默认 torch_env —— Agent 3 主口径
# 「波形→完整流程→出值」实时链依赖 torch/PPGEncoder；实时链不可用时
# chat 层自动降级为查表口径，服务仍可用（PY 覆盖为 base 即降级模式）。
set -e
cd "$(dirname "$0")/.."

PY=${PY:-python}

# Raw dataset root（v3.1+ canonical = data_v2）；override via env
export PPG_DATA_ROOT="${PPG_DATA_ROOT:-<DATA_ROOT>/data_v2}"

echo "=== Starting FastAPI backend (port 8000) ==="
$PY -m uvicorn web.api.main:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!
trap "kill $API_PID 2>/dev/null" EXIT

if [ -d web/frontend/node_modules ]; then
  echo "=== Starting Vite frontend (port 5173) ==="
  cd web/frontend && npm run dev
else
  echo "Frontend not installed yet — run: cd web/frontend && npm install"
  wait $API_PID
fi
