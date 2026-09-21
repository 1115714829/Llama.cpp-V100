#!/bin/bash
# z1-armwatch.sh TAG -- poll one arm; sample GPU memory once the server is healthy.
set -uo pipefail
TAG=$1
LOG=/tmp/z1-$TAG-driver.log
HLOG=/tmp/p60-$TAG.log
SLOG=/tmp/p60-$TAG-server.log
n=0
SEEN=0
while [ $n -lt 120 ]; do
  if [ $SEEN -eq 0 ] && grep -q "health ok=1" "$HLOG" 2>/dev/null; then
    SEEN=1
    echo ===MEM_WHILE_RUNNING=== 
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    echo ===KV_AND_ALLOC_LOG=== 
    grep -a -i -e "KV self size" -e "kv cache" -e "CUDA0 model buffer" -e "out of memory" -e "failed to allocate" -e "CUDA error" "$SLOG" | head -25
  fi
  if grep -q P60_DONE "$LOG" 2>/dev/null; then break; fi
  echo WAIT n=$n time=$(date +%H:%M:%S) last=[$(tail -1 "$HLOG" 2>/dev/null)]
  sleep 15
  n=$((n+1))
done
echo ===DRIVERLOG===
cat "$LOG"
echo ===HARNESSLOG===
cat "$HLOG"
echo ===SERVERLOG_TAIL===
tail -6 "$SLOG"
