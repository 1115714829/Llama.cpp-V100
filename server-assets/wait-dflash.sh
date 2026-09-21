#!/bin/bash
# Wait for / report the draft-dflash test.
set -uo pipefail
L=/tmp/dflash.log
WAIT=${WAIT_S:-330}
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  grep -q 'DFLASH_TEST_DONE' "$L" 2>/dev/null && break
  pgrep -f "[d]flash-test.sh" >/dev/null 2>&1 || break
  sleep 15
done
cat "$L"
echo ""
grep -q 'DFLASH_TEST_DONE' "$L" && echo STATE=DONE || echo STATE=STILL_RUNNING
echo "gpu: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
echo WAITDFLASH_DONE
