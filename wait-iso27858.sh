#!/bin/bash
set -uo pipefail
WAIT=${WAIT_S:-600}
L=/tmp/iso27858.log
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  grep -q 'ISO27858_DONE' "$L" 2>/dev/null && break
  pgrep -f "[p]30-iso27858.sh" >/dev/null 2>&1 || break
  sleep 20
done
cat "$L"
echo ""
grep -q 'ISO27858_DONE' "$L" && echo STATE=DONE || echo STATE=STILL_RUNNING
echo WAITISO_DONE
