#!/bin/bash
set -uo pipefail
L=/tmp/dflash2-official.log
WAIT=${WAIT_S:-500}
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  grep -q 'OFFICIAL_DONE' "$L" 2>/dev/null && break
  pgrep -f "[p]24-dflash2-official.sh" >/dev/null 2>&1 || break
  sleep 15
done
cat "$L"
echo ""
grep -q 'OFFICIAL_DONE' "$L" && echo STATE=DONE || echo STATE=STILL_RUNNING
echo WAITOFFICIAL_DONE
