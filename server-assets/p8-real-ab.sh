#!/bin/bash
# P8: wait for the C4 rebuild, snapshot C4 libs, verify the two lib dirs DIFFER,
# then run a short real A/B (libs loaded via LD_LIBRARY_PATH).
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf

echo "=== [1] wait for build (max 480s) ==="
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do
  n=$((n + 15))
  if [ "$n" -ge 480 ]; then echo "STILL_BUILDING after ${n}s"; break; fi
  sleep 15
  echo "  ... waited ${n}s  | $(tail -1 /tmp/build-c4b.log)"
done
tail -4 /tmp/build-c4b.log
echo ""

echo "=== [2] snapshot C4 libdir ==="
rm -rf /root/libdir-c4
mkdir -p /root/libdir-c4
cp -a "$SRC/build/bin/." /root/libdir-c4/
echo "files: $(ls /root/libdir-c4 | wc -l)"
echo ""

echo "=== [3] VERIFY the two lib dirs really differ (this is the A/B validity check) ==="
for l in libggml-cuda.so.0.24.0 libllama.so.0.4.1 libllama-bench-impl.so; do
  a=$(md5sum "/root/libdir-pristine/$l" 2>/dev/null | awk '{print $1}')
  b=$(md5sum "/root/libdir-c4/$l" 2>/dev/null | awk '{print $1}')
  if [ "$a" = "$b" ]; then v="SAME  <-- BAD, A/B would be meaningless"; else v="DIFFER  <-- good"; fi
  echo "$l  pristine=$a  c4=$b  => $v"
done
echo ""

echo "=== [4] confirm LD_LIBRARY_PATH wins over DT_RUNPATH ==="
echo "--- default resolution (no env) ---"
ldd /root/libdir-pristine/llama-bench 2>/dev/null | grep -E "ggml-cuda|libllama\.so"
echo "--- with LD_LIBRARY_PATH=/root/libdir-c4 ---"
LD_LIBRARY_PATH=/root/libdir-c4 ldd /root/libdir-pristine/llama-bench 2>/dev/null | grep -E "ggml-cuda|libllama\.so"
echo ""

echo "=== [5] short real A/B (plain config, 1 round) ==="
export CUDA_VISIBLE_DEVICES=0
for v in pristine c4; do
  L=/tmp/real-ab-${v}.log
  echo "### variant=$v  LD_LIBRARY_PATH=/root/libdir-$v"
  LD_LIBRARY_PATH=/root/libdir-$v /root/libdir-$v/llama-bench \
    -m "$M" -p 512 -n 128 -ngl -1 -r 3 > "$L" 2>&1
  echo "exit=$?"
  grep -F -e "pp512" -e "tg128" "$L"
  echo ""
done
echo P8_DONE
