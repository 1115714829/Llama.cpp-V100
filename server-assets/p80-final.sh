#!/bin/bash
# Final consolidated measurement (WITH drop_caches, per the harness rule) + all evidence lines.
set -uo pipefail
echo "=== final measurement: tensor, 3 cards (0/1/2), P2P, drop_caches enabled ==="
P2P=1 CARDS=0,1,2 SPLIT=tensor TAG=final-tensor bash /root/p60-ab-harness.sh || true

echo ""
echo "=== /root/goal-baseline.txt (recorded baseline) ==="
cat /root/goal-baseline.txt

echo ""
echo "=== final evidence ==="
grep -a -e "^config:" -e "extra env" -e "libs:" -e "^prompt" -e "^MEDIAN_TG" \
        -e "spec timing:" -e "selector cpu per round" -e "^\[RT\] perf" -e "^greedy:" \
        /tmp/p60-final-tensor.log 2>/dev/null
echo "--- target true cost per round ---"
grep -a "target decode+sync" /tmp/p60-final-tensor-server.log 2>/dev/null | tail -3
echo P80_DONE
