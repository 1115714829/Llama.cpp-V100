#!/bin/bash
# Wait for / report the L3 256k acceptance run.
set -uo pipefail
L=/tmp/l3.log
WAIT=${WAIT_S:-470}
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  grep -q 'L3_DONE' "$L" 2>/dev/null && break
  if ! pgrep -f "[l]3-run.sh" >/dev/null 2>&1; then break; fi
  sleep 20
done

echo "=== markers ==="
grep -E "^##########|^load wait|^request wall|^=== |backend offload|spec-type" "$L" | tail -20
echo ""
echo "=== timings ==="
awk '/^##########/{v=$0} /prompt eval time|eval time =|draft acceptance|total time =/{print v" || "$0}' "$L" | tail -16
echo ""
grep -q 'L3_DONE' "$L" && echo STATE=DONE || echo STATE=STILL_RUNNING
pgrep -f "[l]3-run.sh" >/dev/null 2>&1 && echo runner=running || echo runner=gone
echo "gpu: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
echo WAITL3_DONE
