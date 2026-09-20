#!/bin/bash
# Attribute the remaining ~20 ms/round: the sampler body (set_logits over the full vocabulary).
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
echo "=== [1] rebuild ==="
sed -i 's/\r$//' "$SRC/common/sampling.cpp"
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j82 --target llama-server" > /tmp/build-p72.log 2>&1 &
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do
  sleep 10; n=$((n+10)); [ "$n" -gt 800 ] && { echo "build timeout"; break; }
done
echo -n "build errors: "; grep -icE "error:|FAILED" /tmp/build-p72.log
cp -a "$SRC/build/bin/." /root/libdir-rt/

echo ""
echo "=== [2] tensor + P2P arm (NODROP) ==="
NODROP=1 P2P=1 CARDS=0,1,2 SPLIT=tensor TAG=sampt-tensor bash /root/p60-ab-harness.sh || true

echo ""
echo "=== [3] sampler cost ==="
grep -a "\[RT\] common_sampler_sample" /tmp/p60-sampt-tensor-server.log | tail -6
echo ""
echo "=== throughput ==="
grep -a -e "^prompt" -e "^MEDIAN_TG" /tmp/p60-sampt-tensor.log | tail -5
echo P72_DONE
