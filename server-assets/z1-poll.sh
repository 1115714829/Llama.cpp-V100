#!/bin/bash
# z1-poll.sh -- wait until the box is quiet (no llama-server, no llama-bench, no fa-correctness gate)
LIMIT=45
n=0
while [ "$n" -lt "$LIMIT" ]; do
  S=$(pgrep -x llama-server | wc -l)
  B=$(pgrep -x llama-bench | wc -l)
  F=$(pgrep -f fa-correctnes[s] | wc -l)
  echo "POLL n=$n srv=$S bench=$B fa=$F time=$(date +%H:%M:%S)"
  if [ "$S" -eq 0 ] && [ "$B" -eq 0 ] && [ "$F" -eq 0 ]; then
    echo QUIET
    exit 0
  fi
  sleep 60
  n=$((n+1))
done
echo STILL_BUSY
exit 1
