#!/bin/bash
# Wait for the MTP-off L3 decode test and report.
set -uo pipefail
L=/tmp/l3dec3.log
WAIT=${WAIT_S:-430}
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  grep -q 'L3DEC3_DONE' "$L" 2>/dev/null && break
  pgrep -f "[l]3-decode3.sh" >/dev/null 2>&1 || break
  sleep 15
done
echo "=== markers ==="
grep -E "^##########|^health|^rep[0-9]|^=== |spec-type" "$L" | tail -20
echo ""
echo "=== all eval-time lines (MTP off => should be repeatable) ==="
grep -E "eval time =" "$L" | tail -10
echo ""
grep -q 'L3DEC3_DONE' "$L" && echo STATE=DONE || echo STATE=STILL_RUNNING
echo "gpu: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
echo WAITL3DEC3_DONE
