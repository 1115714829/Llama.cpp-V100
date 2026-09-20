#!/bin/bash
# Pin the draft model to a single device (target stays TP3+tensor+P2P): does it recover ~10 ms/round?
set -uo pipefail
echo "=== arm A: draft on one device (CUDA0) ==="
NODROP=1 P2P=1 CARDS=0,1,2 SPLIT=tensor DEVD=CUDA0 TAG=devd-tensor bash /root/p60-ab-harness.sh || true

echo ""
echo "=== arm B: draft on its own device (CUDA2), target TP3 ==="
NODROP=1 P2P=1 CARDS=0,1,2 SPLIT=tensor DEVD=CUDA2 TAG=devd2-tensor bash /root/p60-ab-harness.sh || true

echo ""
echo "=== comparison ==="
for t in devd-tensor devd2-tensor; do
  echo "--- $t"
  grep -a -e "draft placement" -e "^prompt" -e "^MEDIAN_TG" -e "spec timing:" -e "^\[RT\] perf" /tmp/p60-$t.log 2>/dev/null
  echo ""
done
echo "reference (draft follows tensor split):"
grep -a "^MEDIAN_TG" /tmp/p60-ckpt-tensor.log 2>/dev/null
echo P75_DONE
