#!/bin/bash
set -uo pipefail
L=/tmp/dflash2.log
WAIT=${WAIT_S:-340}
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  grep -q 'DFLASH2_DONE' "$L" 2>/dev/null && break
  pgrep -f "[d]flash-test2.sh" >/dev/null 2>&1 || break
  sleep 15
done
cat "$L"
echo ""
grep -q 'DFLASH2_DONE' "$L" && echo STATE=DONE || echo STATE=STILL_RUNNING
echo "gpu: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
echo WAITDFLASH2_DONE
