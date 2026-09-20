#!/bin/bash
set -uo pipefail
L=/tmp/dflash2-nmax.log
WAIT=${WAIT_S:-460}
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  grep -q 'DF2_NMAX_DONE' "$L" 2>/dev/null && break
  pgrep -f "[p]23-dflash2-nmax.sh" >/dev/null 2>&1 || break
  sleep 15
done
cat "$L"
echo ""
grep -q 'DF2_NMAX_DONE' "$L" && echo STATE=DONE || echo STATE=STILL_RUNNING
echo WAITDF2_DONE
