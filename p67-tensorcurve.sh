#!/bin/bash
# Tensor-mode M-curve with NODROP (fast): establishes the trustworthy M=8 target cost for comparison.
set -uo pipefail
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf

echo "=== tensor M=1 (none) ==="
NODROP=1 CARDS=0,1,2 SPLIT=tensor TAG=ten-none SPEC="--spec-type none" bash /root/p60-ab-harness.sh || true

echo ""
echo "=== tensor M=8 (dflash n=7) ==="
NODROP=1 CARDS=0,1,2 SPLIT=tensor TAG=ten-n7 SPEC="--model-draft $D --spec-type draft-dflash --spec-draft-n-max 7" bash /root/p60-ab-harness.sh || true

echo ""
echo "=== summary ==="
for t in ten-none ten-n7; do
  echo "--- $t"
  grep -a -e "^prompt" -e "^MEDIAN_TG" -e "spec timing:" -e "^\[RT\] perf" /tmp/p60-$t.log 2>/dev/null
  echo ""
done
echo P67_DONE
