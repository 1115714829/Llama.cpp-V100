#!/bin/bash
# z1-dwait.sh TAG -- poll one depth arm until DEPTH_DONE.
set -uo pipefail
TAG=$1
LOG=/tmp/z1-depth-$TAG.log
n=0
while [ $n -lt 150 ]; do
  if grep -q DEPTH_DONE "$LOG" 2>/dev/null; then break; fi
  echo WAIT n=$n time=$(date +%H:%M:%S)
  sleep 20
  n=$((n+1))
done
echo ===DEPTHLOG===
cat "$LOG"
