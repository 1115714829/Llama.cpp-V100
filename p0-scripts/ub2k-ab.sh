#!/bin/bash
# R300: UB=2048 x bucketing stack test. Bars: TTFT vs 208.9 s (r=1.08, ub512);
# extrapolated target ~165 s. Regression bars: tg/track with spec-off pure step
# (34.4 ms anchor), VRAM envelope, no OOM (R295 had 1.7GB mask OOM at SLOTS=3).
set -u
run() {
  tag=$1
  ub=$2
  dec=$3
  slots=$4
  echo "=== ARM $tag UB=$ub decomp=$dec SLOTS=$slots $(date) ==="
  pkill -f 'llama-serve[r]' 2>/dev/null
  pkill -f 'bl2-run.s[h]' 2>/dev/null
  sleep 3
  GGML_KV_BUCKET_RATIO=1.08 LLAMA_SM70_FA_DECOMP=$dec LLAMA_ROUND_TIMING=1 \
    UB=$ub LIBS=/root/libdir-gb SLOTS=$slots NO_SPEC=1 \
    nohup bash /root/llm/test/bl2-run.sh > /root/llm/test/ub-$tag.log 2>&1 &
  for i in $(seq 1 40); do
    code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8082/health)
    if [ "$code" = "200" ]; then echo "health up ${i}x5s"; break; fi
    sleep 5
  done
  python3 /root/llm/test/stress_client.py --base-url http://127.0.0.1:8082 \
    --model Qwen3.8-27B-Q8_0-BL2 --prompt-tokens 235930 \
    --prompt-file /root/llm/test/bl-prompt90.txt --gen 128 --reps 2 --tag $tag \
    > /root/llm/test/ub-$tag-stress.log 2>&1
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -3
  grep -aE 'perf: ctx=Qwen3.8-27B ' /root/llm/test/ub-$tag.log | tail -1
}
run U2K 2048 0 3
run U512 512 0 3
echo UB_DONE
