#!/bin/bash
set -uo pipefail
L=/tmp/mtp-sweep.log
WAIT=${WAIT_S:-540}
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  grep -q 'MTP_SWEEP_DONE' "$L" 2>/dev/null && break
  pgrep -f "[m]tp-sweep.sh" >/dev/null 2>&1 || break
  sleep 15
done
echo "=== per-variant results (3 prompts each) ==="
awk '/^##########/{v=$0} /^\[|draft acceptance/{print v" || "$0}' "$L" | tail -60
echo ""
grep -q 'MTP_SWEEP_DONE' "$L" && echo STATE=DONE || echo STATE=STILL_RUNNING
pgrep -f "[m]tp-sweep.sh" >/dev/null 2>&1 && echo runner=running || echo runner=gone
echo WAITMTP2_DONE
