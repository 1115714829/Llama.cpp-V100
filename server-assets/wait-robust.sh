#!/bin/bash
set -uo pipefail
L=/tmp/prompt-robust.log
WAIT=${WAIT_S:-330}
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  grep -q 'PROMPT_ROBUST_DONE' "$L" 2>/dev/null && break
  pgrep -f "[p]rompt-robust.sh" >/dev/null 2>&1 || break
  sleep 15
done
cat "$L"
echo ""
grep -q 'PROMPT_ROBUST_DONE' "$L" && echo STATE=DONE || echo STATE=STILL_RUNNING
echo WAITROBUST_DONE
