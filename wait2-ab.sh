#!/bin/bash
# Wait for the SECOND run (c4) of the 256k A/B, then stop the remaining rounds.
# Rationale: tg at 256k (~35.63) is ~= tg at short ctx (36.32) for this hybrid model,
# so the extra 2 rounds cost ~46 min for little marginal information; the decisive
# interleaved C4 evidence is the L0 measurement (+2.8%).
set -uo pipefail
L=/tmp/big-ab-256k.log
WAIT=${WAIT_S:-400}
DEADLINE=$(( $(date +%s) + WAIT ))

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  n=$(grep -cF -e "pp262144" "$L" 2>/dev/null || true)
  [ -z "$n" ] && n=0
  if [ "$n" -ge 2 ]; then break; fi
  if ! pgrep -f "[l]lama-bench" >/dev/null 2>&1; then break; fi
  sleep 20
done

n=$(grep -cF -e "pp262144" "$L" 2>/dev/null || true)
[ -z "$n" ] && n=0
echo "runs completed (pp rows): $n"

if [ "$n" -ge 2 ]; then
  echo ">>> two runs done -> stopping the background A/B (rounds 3-4 skipped on purpose)"
  pkill -f "big-ab-256k.sh" 2>/dev/null || true
  sleep 2
  pkill -f "[l]lama-bench" 2>/dev/null || true
  sleep 5
fi

echo ""
echo "=== state at $(date -Is) ==="
echo "--- run markers ---"
grep -E "^(===|--- )" "$L" | tail -8
echo "--- all timing rows ---"
grep -F -e "pp262144" -e "tg128" "$L"
echo ""
echo "--- residual processes ---"
pgrep -f "[l]lama-bench" >/dev/null 2>&1 && echo "bench STILL RUNNING" || echo "bench not running"
echo "--- GPU state ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo WAIT2_DONE
