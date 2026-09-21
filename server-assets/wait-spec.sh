#!/bin/bash
# Wait for / report the speculative-decoding sweep.
set -uo pipefail
L=/tmp/spec-sweep.log
WAIT=${WAIT_S:-430}
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  grep -q 'SPEC_SWEEP_DONE' "$L" 2>/dev/null && break
  pgrep -f "[s]pec-sweep.sh" >/dev/null 2>&1 || break
  sleep 15
done
echo "=== variants and outcomes ==="
awk '/^##########/{v=$0} /prompt eval time|eval time =|draft acceptance|n_gen =|health ok|warn_lines/{print v" || "$0}' "$L" | tail -40
echo ""
grep -q 'SPEC_SWEEP_DONE' "$L" && echo STATE=DONE || echo STATE=STILL_RUNNING
echo "gpu: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
echo WAITSPEC_DONE
