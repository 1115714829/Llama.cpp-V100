#!/bin/bash
# A4: FA KV split floor sweep at long context (diagnostic, no drop_caches to keep the model cached).
while ! grep -q RESTORE2_DONE /tmp/restore2.log 2>/dev/null; do sleep 20; done
grep -a -e RESTORE2_RC -e 'error lines' -e marker /tmp/restore2.log | head -5
export LD_LIBRARY_PATH=/root/libdir-instr
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
LB=/root/llm/test/v100-opt/llama.cpp/build-instr/bin/llama-bench
for f in 0 64 256 1024; do
  echo "--- d=131072 floor=$f"
  GGML_CUDA_FA_SPLIT_FLOOR=$f $LB -m $M -ngl 999 -sm tensor -ts 1/1/1 -fa on -ctk q8_0 -ctv q8_0 -p 0 -n 64 -r 2 -d 131072 2>&1 | grep -a -e tg -e error -e failed | tail -2
done
echo A4_DONE