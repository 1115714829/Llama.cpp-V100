#!/bin/bash
# Rebuild with the process() timer, preserve the baseline snapshot, then re-measure the layer arm
# to attribute the previously unexplained ~57 ms/round.
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp

echo "=== [1] preserve the Goal baseline snapshot ==="
rm -rf /root/libdir-goal-base
cp -a /root/libdir-rt /root/libdir-goal-base
echo -n "baseline snapshot libllama-common md5: "
md5sum /root/libdir-goal-base/libllama-common.so.0.4.1 | awk '{print $1}'

echo ""
echo "=== [2] rebuild ==="
sed -i 's/\r$//' "$SRC/common/speculative.cpp"
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j82 --target llama-server" > /tmp/build-p62.log 2>&1 &
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do
  sleep 10; n=$((n+10)); [ "$n" -gt 800 ] && { echo "build timeout"; break; }
done
echo -n "build errors: "; grep -icE "error:|FAILED" /tmp/build-p62.log
echo -n "new markers in libllama-common: "; grep -c -a "spec overhead per round" "$SRC/build/bin/libllama-common.so.0.4.1"

echo ""
echo "=== [3] re-snapshot to /root/libdir-rt ==="
cp -a "$SRC/build/bin/." /root/libdir-rt/
echo -n "new libllama-common md5: "
md5sum /root/libdir-rt/libllama-common.so.0.4.1 | awk '{print $1}'

echo ""
echo "=== [4] re-measure the working baseline arm (layer) ==="
CARDS=0,1,2 SPLIT=layer TAG=rt-layer bash /root/p60-ab-harness.sh || true

echo ""
echo "=== [5] new attribution ==="
grep -a -e "^MEDIAN_TG" -e "^prompt" -e "spec overhead per round" -e "spec timing:" -e "^\[RT\] perf" -e "greedy:" /tmp/p60-rt-layer.log | tail -20
echo P62_DONE
