#!/bin/bash
# Wait for the corrected decode test (background) and report.
set -uo pipefail
L=/tmp/l3dec2.log
WAIT=${WAIT_S:-430}
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  grep -q 'L3DEC2_DONE' "$L" 2>/dev/null && break
  pgrep -f "[l]3-decode2.sh" >/dev/null 2>&1 || break
  sleep 15
done
echo "=== markers ==="
grep -E "^##########|^health|^resp_bytes|^=== " "$L" | tail -16
echo ""
echo "=== timings ==="
awk '/^##########/{v=$0} /prompt eval time|eval time =|draft acceptance/{print v" || "$0}' "$L" | tail -12
echo ""
grep -q 'L3DEC2_DONE' "$L" && echo STATE=DONE || echo STATE=STILL_RUNNING
echo "gpu: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
echo WAITL3DEC2_DONE
