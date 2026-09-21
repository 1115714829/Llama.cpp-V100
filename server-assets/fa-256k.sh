#!/bin/bash
# 256K prefill A/B (FA change) - q8_0 KV so that 256K fits on TP3.
cd /root
SUM=/tmp/fa-256k.txt
: > $SUM
LB=/root/libdir-instr/llama-bench
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
for pair in pre:/root/libdir-prefa post:/root/libdir-instr; do
  tag=${pair%%:*}; lib=${pair##*:}
  echo "=== 256K $tag lib=$lib ===" >> $SUM
  date >> $SUM
  LD_LIBRARY_PATH=$lib CUDA_VISIBLE_DEVICES=0,1,2 $LB -m $M -ngl 999 -sm tensor -ts 1/1/1 -p 262144 -n 0 -r 1 -ctk q8_0 -ctv q8_0 -fa 1 -ub 2048 -o md > /tmp/fa256-$tag.log 2>&1
  grep -a pp /tmp/fa256-$tag.log >> $SUM
  grep -a -i -e error -e failed /tmp/fa256-$tag.log | tail -2 >> $SUM
done
echo FA256_DONE >> $SUM
