#!/bin/bash
# Establish the Goal baseline on the unmodified tree: layer and tensor, 3 cards same NUMA.
set -uo pipefail
sed -i 's/\r$//' /root/p60-ab-harness.sh

echo "=== arm 1/2: CARDS=0,1,2 SPLIT=layer ==="
CARDS=0,1,2 SPLIT=layer TAG=base-layer bash /root/p60-ab-harness.sh || true

echo ""
echo "=== arm 2/2: CARDS=0,1,2 SPLIT=tensor ==="
CARDS=0,1,2 SPLIT=tensor TAG=base-tensor bash /root/p60-ab-harness.sh || true

echo ""
{
  echo "=== goal-baseline: /root/goal-baseline.txt ==="
  echo "tree: b11053 + C4 + C5 + PR#27858 + instrumentation (unmodified w.r.t. this Goal)"
  echo "date: $(date)"
  echo "fixed: target Qwen3.8-27B-Q8_0.gguf  draft DFlash2 n=7  ctx 8192  official thinking sampling  seed 42  n_predict 512"
  echo ""
  for t in base-layer base-tensor; do
    echo "--- arm $t"
    grep -a -e "^config:" -e "^prompt" -e "^MEDIAN_TG" -e "^greedy:" -e "^libs:" /tmp/p60-$t.log 2>/dev/null
    echo ""
  done
} | tee /root/goal-baseline.txt

echo "P61_DONE"
