#!/bin/bash
# z1-quiet2.sh -- wait until no llama-server, no llama-bench, no fpd chain is running.
set -uo pipefail
LIMIT=80
n=0
while [ $n -lt $LIMIT ]; do
  S=$(pgrep -f 'llama-serve[r] --model' | wc -l)
  B=$(pgrep -f 'llama-benc[h] -m' | wc -l)
  C=$(pgrep -f 'fpd2.-chain.s[h]' | wc -l)
  C2=$(pgrep -f 'fpd-chain.s[h]' | wc -l)
  echo "QUIETCHK n=$n time=$(date +%H:%M:%S) srv=$S bench=$B fpd=$C fpd2=$C2"
  if [ $S -eq 0 ] && [ $B -eq 0 ] && [ $C -eq 0 ] && [ $C2 -eq 0 ]; then
    echo QUIET2_OK
    exit 0
  fi
  sleep 30
  n=$((n+1))
done
echo QUIET2_TIMEOUT
exit 1
