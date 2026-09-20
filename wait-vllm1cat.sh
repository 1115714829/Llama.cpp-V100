#!/bin/bash
set -uo pipefail
WAIT=${WAIT_S:-600}
L=/tmp/vllm1cat-bench.log
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  grep -q 'VLLMBENCH_DONE' "$L" 2>/dev/null && break
  pgrep -f "[p]32-vllm1cat-bench.sh" >/dev/null 2>&1 || break
  sleep 20
done
cat "$L"
echo ""
grep -q 'VLLMBENCH_DONE' "$L" && echo STATE=DONE || echo STATE=STILL_RUNNING
echo WAITVLLM_DONE
