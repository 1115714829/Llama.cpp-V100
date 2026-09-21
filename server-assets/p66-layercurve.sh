#!/bin/bash
# Batch-size scaling IN THE WORKING CONFIG (layer, 3 cards): is the round limited by the target
# forward at M=8 (=> HMMA helps) or by something invariant in M (=> look elsewhere)?
# NODROP=1 so each arm is ~1-2 min instead of ~6 min.
set -uo pipefail
L=/root/libdir-rt
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf

echo "=== any nsys reports left over? ==="
ls -l /tmp/nsys* /root/nsys* 2>&1 | head -5

echo ""
echo "=== arm M=1 (no speculation) ==="
NODROP=1 CARDS=0,1,2 SPLIT=layer TAG=lay-none SPEC="--spec-type none" bash /root/p60-ab-harness.sh || true

echo ""
echo "=== arm M=2 (dflash n=1) ==="
NODROP=1 CARDS=0,1,2 SPLIT=layer TAG=lay-n1 SPEC="--model-draft $D --spec-type draft-dflash --spec-draft-n-max 1" bash /root/p60-ab-harness.sh || true

echo ""
echo "=== arm M=8 (dflash n=7, reference) ==="
NODROP=1 CARDS=0,1,2 SPLIT=layer TAG=lay-n7 SPEC="--model-draft $D --spec-type draft-dflash --spec-draft-n-max 7" bash /root/p60-ab-harness.sh || true

echo ""
echo "=== scaling summary (layer, 3 cards) ==="
for t in lay-none lay-n1 lay-n7; do
  echo "--- $t"
  grep -a -e "^prompt" -e "^MEDIAN_TG" -e "spec timing:" -e "^\[RT\] perf" /tmp/p60-$t.log 2>/dev/null
  echo ""
done
echo P66_DONE
