#!/bin/bash
# z1-wait.sh TAG -- poll one arm until P60_DONE appears in its driver log.
set -uo pipefail
TAG=$1
LOG=/tmp/z1-$TAG-driver.log
HLOG=/tmp/p60-$TAG.log
n=0
while [ $n -lt 60 ]; do
  if grep -q P60_DONE "$LOG" 2>/dev/null; then break; fi
  echo WAIT n=$n time=$(date +%H:%M:%S) last=[$(tail -1 "$HLOG" 2>/dev/null)]
  sleep 30
  n=$((n+1))
done
echo ===DRIVERLOG===
cat "$LOG"
echo ===HARNESSLOG===
cat "$HLOG"
echo ===SERVERLOG_TAIL===
tail -6 /tmp/p60-$TAG-server.log
echo ===GPUMEM=== 
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
