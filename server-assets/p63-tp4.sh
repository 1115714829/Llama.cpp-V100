#!/bin/bash
# Test the TP4 requirement the user pointed at: 4 cards, both split modes, vs the 3-card baseline.
set -uo pipefail
echo "=== TP4 arm A: CARDS=0,1,2,3 SPLIT=tensor ==="
CARDS=0,1,2,3 SPLIT=tensor TAG=tp4-tensor bash /root/p60-ab-harness.sh || true

echo ""
echo "=== TP4 arm B: CARDS=0,1,2,3 SPLIT=layer ==="
CARDS=0,1,2,3 SPLIT=layer TAG=tp4-layer bash /root/p60-ab-harness.sh || true

echo ""
echo "=== summary ==="
for t in tp4-tensor tp4-layer; do
  echo "--- $t"
  grep -a -e "^config:" -e "^prompt" -e "^MEDIAN_TG" -e "spec overhead per round" -e "spec timing:" -e "^\[RT\] perf" -e "^greedy:" /tmp/p60-$t.log 2>/dev/null
  echo ""
done
echo "=== 3-card baseline for reference ==="
grep -a -e "^MEDIAN_TG" /tmp/p60-base-layer.log /tmp/p60-rt-layer.log 2>/dev/null
echo P63_DONE
