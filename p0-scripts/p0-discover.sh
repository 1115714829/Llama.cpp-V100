#!/bin/bash
# P-P0 discovery: read-only environment survey (no writes, no builds).
echo "=== date ==="
date
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
echo "=== gpu procs (llama/bench) ==="
ps -eo pid,comm,args | grep -iE 'llama|bench|vllm' | grep -v grep | head -20
echo "=== gguf under /mnt/3.84t ==="
find /mnt/3.84t -maxdepth 4 -name '*.gguf' -printf '%s %p\n' 2>/dev/null | head -20
echo "=== llama-bench bins ==="
ls -la /root/llm/test/v100-opt/llama.cpp/build*/bin/llama-bench 2>/dev/null
echo "=== libdirs ==="
ls -d /root/libdir-* 2>/dev/null
for d in /root/libdir-pathb /root/libdir-t8b /root/libdir-sm70 /root/libdir-instr; do
  echo "-- $d"
  ls "$d" 2>/dev/null | head -10
  for so in "$d"/libggml-cuda.so* "$d"/libggml-base.so* "$d"/libllama.so*; do
    [ -f "$so" ] || continue
    [ -L "$so" ] && continue
    echo "   $(basename "$so") MD5=$(md5sum "$so" | cut -c1-8) D256=$(strings "$so" | grep -c LLAMA_SM70_D256) SLOTS=$(strings "$so" | grep -c GGML_GALLOCR_SLOTS) Q8D=$(strings "$so" | grep -c LLAMA_SM70_Q8_DIRECT) FAGEMM=$(strings "$so" | grep -c LLAMA_SM70_FA_GEMM)"
  done
done
echo "=== free/disk ==="
free -g | head -2
df -h /mnt/3.84t | tail -1
