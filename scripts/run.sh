#!/bin/bash
# model-bench — start the web UI detached. Stop with stop.sh.
cd "$(dirname "$0")/.." || exit 1
PORT="${PORT:-8090}"
if pgrep -f "[s]erver.py --port $PORT" >/dev/null 2>&1; then
  echo "already running on port $PORT"
  exit 0
fi
PY="${MODEL_BENCH_PY:-}"
if [ -z "$PY" ]; then
  if [ -x /home/eamon/Strata/.venv/bin/python ]; then PY=/home/eamon/Strata/.venv/bin/python; else PY=python3; fi
fi
mkdir -p runs
setsid bash -c "'$PY' server.py --port $PORT --bind 0.0.0.0 >> runs/server.log 2>&1 < /dev/null" < /dev/null > /dev/null 2>&1 &
sleep 1
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "model-bench started: http://${IP:-127.0.0.1}:$PORT  (log: runs/server.log, python: $PY)"
