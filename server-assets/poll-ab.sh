#!/bin/bash
# Poll the background 256k A/B.
set -uo pipefail
L=/tmp/big-ab-256k.log
echo "=== markers ==="
grep -E "^(===|--- )" "$L" | tail -10
echo ""
echo "=== timing rows so far ==="
grep -F -e "pp262144" -e "tg128" "$L" | tail -6
echo ""
if pgrep -f "[l]lama-bench" >/dev/null 2>&1; then echo "STATE=RUNNING"; else echo "STATE=DONE"; fi
echo "now: $(date -Is)"
echo "ALL DONE marker: $(grep -c 'ALL DONE' "$L")"
