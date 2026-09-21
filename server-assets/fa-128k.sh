#!/bin/bash
cd /root
SUM=/tmp/fa-128k.txt
: > $SUM
LB=/root/libdir-instr/llama-bench
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
for pair in pre:/root/libdir-prefa post:/root/libdir-instr; do
  tag=${pair%%:*}; lib=${pair##*:}
  echo "=== 128K $tag lib=$lib ===" >> $SUM
  LD_LIBRARY_PATH=$lib CUDA_VISIBLE_DEVICES=0,1,2 $LB -m $M -ngl 999 -sm tensor -ts 1/1/1 -p 131072 -n 0 -r 1 -ctk f16 -ctv f16 -fa 1 -ub 2048 -o md > /tmp/fa128-$tag.log 2>&1
  grep -a pp /tmp/fa128-$tag.log >> $SUM
done
echo FA128_DONE >> $SUM
