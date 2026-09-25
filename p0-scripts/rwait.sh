#!/bin/bash
# rwait.sh <name> [timeout_s] [tail_lines]
#
# Blocking wait for a job started with rjob.sh: a 5 s sleep loop until the
# "JOB_DONE_<name>" terminator shows up in /root/llm/test/<name>.log, then print
# the terminator line plus the tail. One SSH call, zero polling, zero context.
#
#   rwait.sh build 1200 40
#
# Exit code: 0 = marker seen, 1 = timeout. LOG=/path overrides the log file.
set -u
NAME=${1:?usage: rwait.sh <name> [timeout_s] [tail_lines]}
TMO=${2:-1800}
TAILN=${3:-25}
LOG=${LOG:-/root/llm/test/$NAME.log}
MARK="JOB_DONE_$NAME"

waited=0
while ! grep -q "$MARK" "$LOG" 2>/dev/null; do
  sleep 5
  waited=$((waited + 5))
  if [ "$waited" -ge "$TMO" ]; then
    echo "WAIT_TIMEOUT name=$NAME waited=${waited}s (job may still run: tmux ls)"
    tail -n "$TAILN" "$LOG" 2>/dev/null
    exit 1
  fi
done

echo "== $MARK after ${waited}s =="
grep -a "$MARK" "$LOG" | tail -1
tail -n "$TAILN" "$LOG"
exit 0
