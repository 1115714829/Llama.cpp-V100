#!/bin/bash
# 1) wait for p78, 2) rebuild ggml-cuda from the restored mmvq.cu (the sweep left it uncompiled),
# 3) A/B CUDA graphs.
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp

echo "=== wait for p78 to finish ==="
n=0
while pgrep -f "[p]78-discriminator" >/dev/null 2>&1; do
  sleep 20; n=$((n+20)); [ "$n" -gt 1500 ] && { echo "p78 timeout"; break; }
done
echo "p78 results:"
sed -n '/=== results ===/,$p' /tmp/p78.out 2>/dev/null

echo ""
echo "=== rebuild ggml-cuda from the restored (base) mmvq.cu ==="
echo -n "ncols 5..8 in source: "
grep -A 14 'MMVQ_PARAMETERS_VOLTA) {' "$SRC/ggml/src/ggml-cuda/mmvq.cu" | sed -n '10,14p' | tr -d '\n' | sed 's/  */ /g'
echo ""
touch "$SRC/ggml/src/ggml-cuda/mmvq.cu"
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j82 --target llama-server" > /tmp/build-p79.log 2>&1 &
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do sleep 10; n=$((n+10)); [ "$n" -gt 900 ] && break; done
echo -n "build errors: "; grep -icE "error:|FAILED" /tmp/build-p79.log
cp -a "$SRC/build/bin/." /root/libdir-rt/
echo -n "libggml-cuda md5 now: "; md5sum /root/libdir-rt/libggml-cuda.so.0.24.0 | awk '{print $1}'
echo "baseline md5 was     : b4b467736ca4a58ac1c747c7ff14fb00"

echo ""
echo "=== A/B: CUDA graphs ON (default) ==="
NODROP=1 P2P=1 CARDS=0,1,2 SPLIT=tensor TAG=graph-on bash /root/p60-ab-harness.sh || true

echo ""
echo "=== A/B: CUDA graphs DISABLED ==="
NODROP=1 P2P=1 NOGRAPH=1 CARDS=0,1,2 SPLIT=tensor TAG=graph-off bash /root/p60-ab-harness.sh || true

echo ""
echo "=== graph A/B result ==="
for t in graph-on graph-off; do
  echo "--- $t"
  grep -a -e "extra env" -e "^prompt" -e "^MEDIAN_TG" -e "spec timing:" -e "libggml-cuda" /tmp/p60-$t.log 2>/dev/null
  grep -a "target decode+sync" /tmp/p60-$t-server.log 2>/dev/null | tail -3
  echo ""
done
echo P79_DONE
