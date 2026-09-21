#!/bin/bash
# Wait (bounded) for the background 256k A/B, then report. Keeps one turn per ~9 min instead of ~1 min.
set -uo pipefail
L=/tmp/big-ab-256k.log
WAIT=${WAIT_S:-520}
DEADLINE=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if grep -q 'ALL DONE' "$L" 2>/dev/null; then break; fi
  if ! pgrep -f "[l]lama-bench" >/dev/null 2>&1; then break; fi
  sleep 20
done

echo "=== state at $(date -Is) ==="
echo "--- run markers ---"
grep -E "^(===|--- )" "$L" | tail -8
echo "--- timing rows so far ---"
grep -F -e "pp262144" -e "tg128" "$L" | tail -8
echo ""
if grep -q 'ALL DONE' "$L"; then
  echo "STATE=DONE"
  echo "--- full summary ---"
  grep -F -e "pp262144" -e "tg128" "$L"
else
  echo "STATE=STILL_RUNNING"
  ps -eo pid,etime,comm | grep -E "llama-bench" | grep -v grep || echo "(no bench process)"
fi
echo WAIT_AB_DONE
