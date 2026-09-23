#!/bin/bash
# P-P3 perf A/B (staged-baseline protocol): ABBA 4x 256K stress, same build/env
# gate (only LLAMA_SM70_FA_DECOMP flips). Baseline = R297 adopted 208.9 s.
set -u
run() {
  tag=$1
  dec=$2
  echo "=== AB ARM $tag decomp=$dec $(date) ==="
  pkill -f 'bl2-run.s[h]' 2>/dev/null
  pkill -f 'llama-serve[r]' 2>/dev/null
  sleep 3
  GGML_KV_BUCKET_RATIO=1.08 LLAMA_SM70_FA_DECOMP=$dec LLAMA_ROUND_TIMING=1 \
    LIBS=/root/libdir-gb SLOTS=3 nohup bash /root/llm/test/bl2-run.sh > /root/llm/test/ab-$tag.log 2>&1 &
  for i in $(seq 1 40); do
    code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8082/health)
    if [ "$code" = "200" ]; then echo "health up ${i}x5s"; break; fi
    sleep 5
  done
  python3 /root/llm/test/stress_client.py --base-url http://127.0.0.1:8082 \
    --model Qwen3.8-27B-Q8_0-BL2 --prompt-tokens 235930 \
    --prompt-file /root/llm/test/bl-prompt90.txt --gen 128 --reps 2 --tag $tag \
    > /root/llm/test/ab-$tag-stress.log 2>&1
  echo "=== RT $tag ==="
  grep -aE 'perf: ctx=Qwen3.8-27B ' /root/llm/test/ab-$tag.log | tail -1
}
run C1 0
run F1 1
run F2 1
run C2 0
echo AB_DONE
