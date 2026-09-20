#!/bin/bash
# Wait for the corrected decode test (l3-decode.sh) to finish and report.
set -uo pipefail
L=/tmp/l3-dec.log
WAIT=${WAIT_S:-470}
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  grep -q 'L3DEC_DONE' "$L" 2>/dev/null && break
  pgrep -f "[l]3-decode.sh" >/dev/null 2>&1 || break
  sleep 20
done
echo "=== markers ==="
grep -E "^##########|^load wait|^resp bytes|^usage|^finish_reason|^content_len|^=== " "$L" | tail -20
echo ""
echo "=== timings ==="
awk '/^##########/{v=$0} /prompt eval time|eval time =|draft acceptance/{print v" || "$0}' "$L" | tail -12
echo ""
grep -q 'L3DEC_DONE' "$L" && echo STATE=DONE || echo STATE=STILL_RUNNING
pgrep -f "[l]3-decode.sh" >/dev/null 2>&1 && echo runner=running || echo runner=gone
echo "gpu: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
echo WAITL3DEC_DONE
