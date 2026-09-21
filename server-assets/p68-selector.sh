#!/bin/bash
# Verify the parallel CPU selector: cost drop + bit-identical candidates (greedy hash unchanged).
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
BASE_HASH=69207026ca43b19f5df2871a658d9f88bf3c583387a39d4a5ecceb8b21eaa147

echo "=== [1] rebuild ==="
sed -i 's/\r$//' "$SRC/common/speculative.cpp"
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j82 --target llama-server" > /tmp/build-p68.log 2>&1 &
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do
  sleep 10; n=$((n+10)); [ "$n" -gt 800 ] && { echo "build timeout"; break; }
done
echo -n "build errors: "; grep -icE "error:|FAILED" /tmp/build-p68.log
echo -n "new marker: "; grep -c -a "selector cpu per round" "$SRC/build/bin/libllama-common.so.0.4.1"

echo ""
echo "=== [2] snapshot ==="
cp -a "$SRC/build/bin/." /root/libdir-rt/
md5sum /root/libdir-rt/libllama-common.so.0.4.1 | awk '{print $1"  libllama-common (was b5475336ef7b8529bf7654855dcc94de)"}'

echo ""
echo "=== [3] tensor arm with the parallel selector (NODROP=1) ==="
NODROP=1 CARDS=0,1,2 SPLIT=tensor TAG=sel-tensor bash /root/p60-ab-harness.sh || true

echo ""
echo "=== [4] result: cost + correctness ==="
grep -a -e "selector cpu per round" -e "^prompt" -e "^MEDIAN_TG" -e "spec timing:" -e "^greedy:" -e "^\[RT\] perf" /tmp/p60-sel-tensor.log | tail -16
echo ""
echo "baseline tensor greedy hash : $BASE_HASH"
echo -n "this run tensor greedy hash : "
grep -a -o "sha256=[0-9a-f]*" /tmp/p60-sel-tensor.log | tail -1 | cut -d= -f2
echo P68_DONE
