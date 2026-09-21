#!/bin/bash
# Clean 256K decode point with the current library (formal: drop_caches, nothing else running).
cd /root/llm/test/v100-opt/llama.cpp || exit 1
export LD_LIBRARY_PATH=/root/libdir-instr
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
LB=/root/llm/test/v100-opt/llama.cpp/build-instr/bin/llama-bench
sync; echo 3 > /proc/sys/vm/drop_caches
echo '--- d=262144 (M=1 plain decode, q8_0 KV, TP3) ---'
$LB -m $M -ngl 999 -sm tensor -ts 1/1/1 -fa on -ctk q8_0 -ctv q8_0 -p 0 -n 64 -r 2 -d 262144 2>&1 | grep -a -e tg -e error -e failed | tail -3
echo D256_DONE