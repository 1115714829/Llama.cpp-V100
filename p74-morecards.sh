#!/bin/bash
# Does more aggregate bandwidth (more cards, with P2P) beat the all-reduce cost?
# Quant is fixed by the objective, so bytes/card is the only free variable: 4 cards -> 7.25, 6 -> 4.8 GB/card.
set -uo pipefail
echo "=== TP4 tensor + P2P ==="
NODROP=1 P2P=1 CARDS=0,1,2,3 SPLIT=tensor TAG=tp4p2p-tensor bash /root/p60-ab-harness.sh || true

echo ""
echo "=== TP6 tensor + P2P ==="
NODROP=1 P2P=1 CARDS=0,1,2,3,4,5 SPLIT=tensor TAG=tp6p2p-tensor bash /root/p60-ab-harness.sh || true

echo ""
echo "=== TP6 layer + P2P ==="
NODROP=1 P2P=1 CARDS=0,1,2,3,4,5 SPLIT=layer TAG=tp6p2p-layer bash /root/p60-ab-harness.sh || true

echo ""
echo "=== summary ==="
for t in tp4p2p-tensor tp6p2p-tensor tp6p2p-layer; do
  echo "--- $t"
  grep -a -e "^config:" -e "^prompt" -e "^MEDIAN_TG" -e "spec timing:" -e "^\[RT\] perf" /tmp/p60-$t.log 2>/dev/null
  echo ""
done
echo "reference: TP3 tensor + P2P = 78.95 median (from /tmp/p60-ckpt-tensor.log)"
grep -a "^MEDIAN_TG" /tmp/p60-ckpt-tensor.log 2>/dev/null
echo P74_DONE
