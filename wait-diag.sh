#!/bin/bash
set -uo pipefail
L=/tmp/diag1a.log
WAIT=${WAIT_S:-560}
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  grep -q 'DIAG1A_DONE' "$L" 2>/dev/null && break
  pgrep -f "[p]26-diag1a.sh" >/dev/null 2>&1 || break
  sleep 20
done
cat "$L"
echo ""
grep -q 'DIAG1A_DONE' "$L" && echo STATE=DONE || echo STATE=STILL_RUNNING
grep -iE "error:|FAILED" /tmp/build-diag1a.log | head -5 || true
tail -2 /tmp/build-diag1a.log 2>/dev/null
echo WAITDIAG_DONE
