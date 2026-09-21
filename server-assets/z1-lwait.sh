#!/bin/bash
# z1-lwait.sh TAG -- poll one ladder arm until LADDER_DONE.
set -uo pipefail
TAG=$1
LOG=/tmp/z1-ladder-$TAG.log
n=0
while [ $n -lt 200 ]; do
  if grep -q LADDER_DONE "$LOG" 2>/dev/null; then break; fi
  sleep 30
  n=$((n+1))
done
echo ===LADDERLOG===
grep -a -e tg32 -e LADDER_DONE -e BENCH_RC -e PREFLIGHT -e PREFLIGHT_BUSY "$LOG"
