#!/bin/bash
# Measure the synchronous wait hidden in common_sampler_sample (tensor + P2P, NODROP).
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
echo "=== [1] rebuild ==="
sed -i 's/\r$//' "$SRC/common/sampling.cpp"
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j82 --target llama-server" > /tmp/build-p71.log 2>&1 &
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do
  sleep 10; n=$((n+10)); [ "$n" -gt 800 ] && { echo "build timeout"; break; }
done
echo -n "build errors: "; grep -icE "error:|FAILED" /tmp/build-p71.log
echo -n "marker: "; grep -c -a "llama_synchronize = " "$SRC/build/bin/libllama-common.so.0.4.1"
cp -a "$SRC/build/bin/." /root/libdir-rt/

echo ""
echo "=== [2] tensor + P2P arm (NODROP) ==="
NODROP=1 P2P=1 CARDS=0,1,2 SPLIT=tensor TAG=sync-tensor bash /root/p60-ab-harness.sh || true

echo ""
echo "=== [3] the hidden wait ==="
grep -a "\[RT\] common_sampler_sample" /tmp/p60-sync-tensor-server.log | tail -5
echo ""
echo "=== throughput + other phases ==="
grep -a -e "^prompt" -e "^MEDIAN_TG" -e "spec timing:" -e "^\[RT\] perf" /tmp/p60-sync-tensor.log | tail -8
echo ""
echo "=== per-round attribution (from the server log) ==="
grep -a "spec overhead per round" /tmp/p60-sync-tensor-server.log | tail -3
echo P71_DONE
