#!/bin/bash
# Attribute the last ~18 ms: per-round speculative checkpoint save/restore for the hybrid (GDN) model.
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
echo "=== [1] rebuild (server only) ==="
sed -i 's/\r$//' "$SRC/tools/server/server-context.cpp"
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j82 --target llama-server" > /tmp/build-p73.log 2>&1 &
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do
  sleep 10; n=$((n+10)); [ "$n" -gt 800 ] && { echo "build timeout"; break; }
done
echo -n "build errors: "; grep -icE "error:|FAILED" /tmp/build-p73.log
cp -a "$SRC/build/bin/." /root/libdir-rt/

echo ""
echo "=== [2] tensor + P2P arm (NODROP) ==="
NODROP=1 P2P=1 CARDS=0,1,2 SPLIT=tensor TAG=ckpt-tensor bash /root/p60-ab-harness.sh || true

echo ""
echo "=== [3] checkpoint cost ==="
grep -a "\[RT\] spec checkpoints" /tmp/p60-ckpt-tensor-server.log | tail -6
echo ""
echo "=== [4] checkpoint size evidence (debug lines, if any) ==="
grep -a "created speculative checkpoint" /tmp/p60-ckpt-tensor-server.log | tail -3
echo ""
echo "=== [5] throughput ==="
grep -a -e "^prompt" -e "^MEDIAN_TG" /tmp/p60-ckpt-tensor.log | tail -5
echo P73_DONE
