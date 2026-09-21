#!/bin/bash
# Decisive discriminator: how much of the target's M=8 round is all-reduce vs GEMM?
#   - Q8_0 TP2 uses llama.cpp's FAST 2-device all-reduce path (Q8_0 is 29 GB; TP2 = 14.5 GB/card,
#     tight but the objective's quant must stay fixed)
#   - Q2_K_XL (9.14 GB, fits one card) TP1 vs TP3 isolates all-reduce at identical weights
# NODROP for speed; LLAMA_SPEC_TIMING gives the target decode+sync per round.
set -uo pipefail
L=/root/libdir-rt
M2=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
SPEC="--model-draft $D --spec-type draft-dflash --spec-draft-n-max 7"

echo "=== arm 1: Q8_0, TP2 (fast 2-device all-reduce), tensor, P2P ==="
NODROP=1 P2P=1 CARDS=0,1 SPLIT=tensor TAG=tp2-tensor bash /root/p60-ab-harness.sh || true

echo ""
echo "=== arm 2: Q2_K_XL, TP1 (NO all-reduce), tensor ==="
NODROP=1 CARDS=0 SPLIT=tensor TAG=q2tp1-tensor M="$M2" SPEC="$SPEC" bash /root/p60-ab-harness.sh || true

echo ""
echo "=== arm 3: Q2_K_XL, TP3, tensor, P2P (same weights, with all-reduce) ==="
NODROP=1 P2P=1 CARDS=0,1,2 SPLIT=tensor TAG=q2tp3-tensor M="$M2" SPEC="$SPEC" bash /root/p60-ab-harness.sh || true

echo ""
echo "=== results ==="
for t in tp2-tensor q2tp1-tensor q2tp3-tensor; do
  echo "--- $t"
  grep -a -e "^config:" -e "^prompt" -e "^MEDIAN_TG" -e "spec timing:" /tmp/p60-$t.log 2>/dev/null
  grep -a "\[RT\] target decode+sync" /tmp/p60-$t-server.log 2>/dev/null | tail -2
  echo ""
done
echo P78_DONE
