#!/bin/bash
# model-bench — stop the web UI.
PORT="${PORT:-8090}"
PID=$(pgrep -f "[s]erver.py --port $PORT")
if [ -n "$PID" ]; then
  kill $PID && echo "stopped $PID"
else
  echo "not running on port $PORT"
fi
