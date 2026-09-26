#!/bin/bash
# stop-llama.sh [port] - stop the TEST llama-server started by llama-std.sh (matched by --alias sm70-llama).
# Never touches systemd units or any other llama-server. Waits until the process and the port are gone.
PORT=${1:-8090}
PAT='llama-serve[r] .*--alias sm70-llama'
pkill -f "$PAT" 2> /dev/null
for i in $(seq 1 24); do
  if ! pgrep -f "$PAT" > /dev/null && ! (exec 3<> /dev/tcp/127.0.0.1/$PORT) 2> /dev/null; then
    echo "STOPPED port=$PORT after=$((i * 5))s"
    exit 0
  fi
  sleep 5
done
pkill -9 -f "$PAT" 2> /dev/null
sleep 5
if pgrep -f "$PAT" > /dev/null; then echo "STOP_FAILED"; exit 1; fi
echo "STOPPED_KILL9 port=$PORT"
