#!/bin/bash
# Wait for the MTP experiment and report per-variant timings.
set -uo pipefail
L=/tmp/mtp-exp.log
WAIT=${WAIT_S:-470}
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  grep -q 'MTP_EXP_DONE' "$L" 2>/dev/null && break
  if ! pgrep -f "[m]tp-exp.sh" >/dev/null 2>&1 && ! pgrep -f "[l]lama-server" >/dev/null 2>&1; then break; fi
  sleep 20
done

echo "=== per-variant markers ==="
grep -E "^############|^load wait|^offload_failed_lines|^resp_bytes|not supported with" "$L" | tail -30
echo ""
echo "=== timings (variant | line) ==="
awk '/^############ variant=/{v=$0} /prompt eval time|eval time =|draft acceptance|n_gen =/{print v" || "$0}' "$L" | tail -24
echo ""
grep -q 'MTP_EXP_DONE' "$L" && echo STATE=DONE || echo STATE=STILL_RUNNING
echo "gpu: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
echo WAITMTP_DONE
