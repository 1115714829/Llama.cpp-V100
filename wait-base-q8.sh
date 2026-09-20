#!/bin/bash
set -uo pipefail
WAIT=${WAIT_S:-2400}
L=/tmp/base-q8.log
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  grep -q 'BASEQ8_DONE' "$L" 2>/dev/null && break
  pgrep -f "[p]35-base-q8.sh" >/dev/null 2>&1 || break
  sleep 20
done
cat "$L"
echo ""
grep -q 'BASEQ8_DONE' "$L" && echo STATE=DONE || echo STATE=STILL_RUNNING
echo WAITBASEQ8_DONE
