#!/bin/bash
set -uo pipefail
L=/tmp/multigpu.log
WAIT=${WAIT_S:-520}
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  grep -q 'MULTIGPU_DONE' "$L" 2>/dev/null && break
  pgrep -f "[p]25-multigpu.sh" >/dev/null 2>&1 || break
  sleep 15
done
cat "$L"
echo ""
grep -q 'MULTIGPU_DONE' "$L" && echo STATE=DONE || echo STATE=STILL_RUNNING
echo WAITMG_DONE
