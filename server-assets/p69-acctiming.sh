#!/bin/bash
# Locate the remaining ~21 ms/round in tensor mode: per-round draft/accept attribution.
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
echo "=== [1] rebuild ==="
sed -i 's/\r$//' "$SRC/common/speculative.cpp"
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j82 --target llama-server" > /tmp/build-p69.log 2>&1 &
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do
  sleep 10; n=$((n+10)); [ "$n" -gt 800 ] && { echo "build timeout"; break; }
done
echo -n "build errors: "; grep -icE "error:|FAILED" /tmp/build-p69.log
cp -a "$SRC/build/bin/." /root/libdir-rt/

echo ""
echo "=== [2] tensor arm (NODROP) ==="
NODROP=1 CARDS=0,1,2 SPLIT=tensor TAG=acc-tensor bash /root/p60-ab-harness.sh || true

echo ""
echo "=== [3] per-round attribution ==="
grep -a -e "^prompt" -e "^MEDIAN_TG" -e "spec overhead per round" -e "spec timing:" -e "selector cpu per round" -e "^\[RT\] perf" /tmp/p60-acc-tensor.log | tail -14
echo P69_DONE
