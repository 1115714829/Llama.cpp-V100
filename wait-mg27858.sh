#!/bin/bash
# Poll the 3-card PR #27858 matrix.
set -uo pipefail
WAIT=${WAIT_S:-900}
L=/tmp/mg27858.log
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  grep -q 'MG27858_DONE' "$L" 2>/dev/null && break
  pgrep -f "[p]29-mg27858.sh" >/dev/null 2>&1 || break
  sleep 20
done
cat "$L"
echo ""
grep -q 'MG27858_DONE' "$L" && echo STATE=DONE || echo STATE=STILL_RUNNING
echo WAITMG27858_DONE
