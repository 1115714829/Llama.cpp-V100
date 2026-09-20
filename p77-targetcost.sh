#!/bin/bash
# Attribute the target's TRUE per-round cost (decode + completion wait), and confirm the tree is back
# at the base mmvq setting after the nwarps sweep.
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp

echo "=== [0] confirm mmvq.cu is back to the base setting (ncols 5..8 -> 2) ==="
grep -A 14 'MMVQ_PARAMETERS_VOLTA) {' "$SRC/ggml/src/ggml-cuda/mmvq.cu" | sed -n '1,16p'

echo ""
echo "=== [1] rebuild ==="
sed -i 's/\r$//' "$SRC/tools/server/server-context.cpp"
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j82 --target llama-server" > /tmp/build-p77.log 2>&1 &
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do
  sleep 10; n=$((n+10)); [ "$n" -gt 900 ] && { echo "build timeout"; break; }
done
echo -n "build errors: "; grep -icE "error:|FAILED" /tmp/build-p77.log
cp -a "$SRC/build/bin/." /root/libdir-rt/

echo ""
echo "=== [2] reference arm: tensor + P2P, 3 cards (NODROP) ==="
NODROP=1 P2P=1 CARDS=0,1,2 SPLIT=tensor TAG=tgt-tensor bash /root/p60-ab-harness.sh || true

echo ""
echo "=== [3] target true cost (decode + sync) ==="
grep -a "\[RT\] target decode+sync" /tmp/p60-tgt-tensor-server.log | tail -5
echo ""
echo "=== [4] throughput + other phases ==="
grep -a -e "^prompt" -e "^MEDIAN_TG" -e "spec timing:" -e "^\[RT\] perf" /tmp/p60-tgt-tensor.log | tail -8
echo P77_DONE
