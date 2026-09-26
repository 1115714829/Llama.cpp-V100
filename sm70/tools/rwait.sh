#!/bin/bash
# rwait.sh <name> [timeout_s] [tail_lines]
# Block (5 s sleep loop) until "JOB_DONE_<name>" appears in the job log, then print it plus the tail.
# Exit 0 = done, 1 = timeout (the job may still be running: check tmux ls, do not relaunch).
set -u
NAME=${1:?usage: rwait.sh <name> [timeout_s] [tail_lines]}
TMO=${2:-1800}
TAILN=${3:-25}
LOG=${LOG:-/root/llm/test/sm70/logs/$NAME.log}
MARK="JOB_DONE_$NAME"

waited=0
while ! grep -q "$MARK" "$LOG" 2>/dev/null; do
  sleep 5
  waited=$((waited + 5))
  if [ "$waited" -ge "$TMO" ]; then
    echo "WAIT_TIMEOUT name=$NAME waited=${waited}s"
    tail -n "$TAILN" "$LOG" 2>/dev/null
    exit 1
  fi
done
echo "== $MARK after ${waited}s =="
grep -a "$MARK" "$LOG" | tail -1
tail -n "$TAILN" "$LOG"
exit 0
