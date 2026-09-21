#!/bin/bash
# z1-quiet3.sh -- wait for machine idle: no llama-server/bench, no fpd chains, no cmake, no build lock.
set -uo pipefail
LIMIT=90
n=0
while [ $n -lt $LIMIT ]; do
  S=$(pgrep -f 'llama-serve[r] --model' | wc -l)
  B=$(pgrep -f 'llama-benc[h] -m' | wc -l)
  C1=$(pgrep -f 'fpd[-]chain' | wc -l)
  C2=$(pgrep -f 'fpd2[-]chain' | wc -l)
  M=$(pgrep -f 'cmake --buil[d]' | wc -l)
  L=0; [ -e /tmp/LLAMA_BUILD_LOCK ] && L=1
  echo "CHK n=$n t=$(date +%H:%M:%S) srv=$S bench=$B c1=$C1 c2=$C2 cmake=$M lock=$L"
  if [ $S -eq 0 ] && [ $B -eq 0 ] && [ $C1 -eq 0 ] && [ $C2 -eq 0 ] && [ $M -eq 0 ] && [ $L -eq 0 ]; then
    sleep 25
    S2=$(pgrep -f 'llama-serve[r] --model' | wc -l)
    B2=$(pgrep -f 'llama-benc[h] -m' | wc -l)
    C22=$(pgrep -f 'fpd2[-]chain' | wc -l)
    if [ $S2 -eq 0 ] && [ $B2 -eq 0 ] && [ $C22 -eq 0 ]; then echo QUIET3_OK t=$(date +%H:%M:%S); exit 0; fi
  fi
  sleep 30
  n=$((n+1))
done
echo QUIET3_TIMEOUT
exit 1
