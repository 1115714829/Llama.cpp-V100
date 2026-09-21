#!/bin/bash
# (1) tick the per-round attribution from draft() (fixed), (2) test the GGML_CUDA_P2P switch.
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
echo "=== [1] rebuild ==="
sed -i 's/\r$//' "$SRC/common/speculative.cpp"
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j82 --target llama-server" > /tmp/build-p70.log 2>&1 &
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do
  sleep 10; n=$((n+10)); [ "$n" -gt 800 ] && { echo "build timeout"; break; }
done
echo -n "build errors: "; grep -icE "error:|FAILED" /tmp/build-p70.log
cp -a "$SRC/build/bin/." /root/libdir-rt/

echo ""
echo "=== [2] arm A: tensor reference (NODROP) ==="
NODROP=1 CARDS=0,1,2 SPLIT=tensor TAG=ref-tensor bash /root/p60-ab-harness.sh || true

echo ""
echo "=== [3] arm B: tensor + GGML_CUDA_P2P=1 (NODROP) ==="
NODROP=1 P2P=1 CARDS=0,1,2 SPLIT=tensor TAG=p2p-tensor bash /root/p60-ab-harness.sh || true

echo ""
echo "=== [4] comparison ==="
for t in ref-tensor p2p-tensor; do
  echo "--- $t"
  grep -a -e "extra env" -e "^prompt" -e "^MEDIAN_TG" -e "spec overhead per round" -e "spec timing:" -e "selector cpu per round" -e "^\[RT\] perf" /tmp/p60-$t.log 2>/dev/null
  echo ""
done
echo P70_DONE
