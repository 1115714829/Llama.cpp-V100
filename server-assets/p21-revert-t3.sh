#!/bin/bash
# t3 revert: remove the Volta MMQ config experiment, rebuild, and prove the result is byte-identical
# to the previous C4+C5-only library (md5 9902c7c3... = libdir-volta2).
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
G="$SRC/ggml/src/ggml-cuda"

echo "=== [1] revert: drop the volta config file + normalize ==="
rm -f "$G/mmq-config-volta.cuh"
sed -i 's/\r$//' "$G/mmq.cuh"
echo "mmq-config-volta refs in mmq.cuh (expect 0): $(grep -c 'mmq-config-volta' "$G/mmq.cuh")"
echo "file present? $(ls "$G/mmq-config-volta.cuh" 2>/dev/null || echo NO)"
echo ""

echo "=== [2] rebuild ==="
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j8 --target llama-bench llama-server" > /tmp/build-revert.log 2>&1 &
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do
  n=$((n+15)); [ "$n" -ge 440 ] && { echo "still building" | tee -a /dev/null; break; }
  sleep 15
done
grep -iE "error:|FAILED" /tmp/build-revert.log | head -5 || true
tail -2 /tmp/build-revert.log
echo ""

echo "=== [3] verify the revert is byte-exact (same source => same lib) ==="
md5sum "$SRC/build/bin/libggml-cuda.so.0.24.0"
echo "volta2 (C4+C5 only) was: 9902c7c30a23320727c18fdbc0b6fd35"
echo ""

echo "=== [4] restore libdir-volta2 content into the build dir state, and re-check the two wins still hold ==="
rm -rf /root/libdir-final && mkdir -p /root/libdir-final
cp -a "$SRC/build/bin/." /root/libdir-final/
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
export CUDA_VISIBLE_DEVICES=0
for UB in 8 32; do
  ROW=$(LD_LIBRARY_PATH=/root/libdir-final /root/libdir-final/llama-bench -m "$M" -p 512 -n 32 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 -b "$UB" -ub "$UB" -r 3 2>&1 | grep -F pp512)
  printf "final UB=%-3s %s\n" "$UB" "$ROW"
done
echo REVERT_DONE
