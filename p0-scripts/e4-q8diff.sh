#!/bin/bash
# e4-q8diff.sh - same-binary A/B per-op GPU table: staged q8_0->f16 vs in-kernel q8_0.
#
# Arm A: GGML_FA_TILE_Q8_DIRECT=0  (tile kernel stages q8_0 -> f16 mirror each call)
# Arm B: GGML_FA_TILE_Q8_DIRECT=1  (tile kernel dequants q8_0 in-kernel, no mirror)
# Same binary, same env otherwise -> the [OP] table difference attributes the
# -5.2 ms/round win (and its gap vs the -12 ms estimate) to specific ops.
set -u
RUNLOG=/root/llm/test/e4-run.log
: > "$RUNLOG"
OPENV="GGML_CUDA_OP_TIMING=1 GGML_CUDA_DISABLE_GRAPHS=1 GGML_META_SUBGRAPH_CAPTURE=0 GGML_CUDA_NO_CONCURRENT=1"

run() {
  local tag=$1; local q8=$2
  pkill -f 'llama-serve[r] --model' 2>/dev/null
  sleep 4
  tmux kill-session -t srv 2>/dev/null
  tmux new-session -d -s srv "TAG=$tag SPEC=1 ROUND=1 UB=2048 GGML_FA_TILE_Q8_DIRECT=$q8 $OPENV bash /root/llm/test/t1c-run.sh"
  local code=000
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8082/health)
    [ "$code" = "200" ] && break
    sleep 5
  done
  echo "HEALTH $tag code=$code q8=$q8" >> "$RUNLOG"
  if [ "$code" != "200" ]; then
    tail -4 /root/llm/test/$tag.log >> "$RUNLOG"
    return 1
  fi
  python3 /root/llm/test/stress_client.py --base-url http://127.0.0.1:8082 \
    --model Qwen3.8-27B-Q8_0-BL2 --prompt-tokens 235930 --prompt-file bl-prompt90.txt \
    --gen 320 --reps 1 --tag $tag --timeout 1800 >> "$RUNLOG" 2>&1
  sleep 3
  pkill -f 'llama-serve[r] --model' 2>/dev/null
  sleep 4
  echo "=== $tag [OP] last window (q8_direct=$q8) ===" >> "$RUNLOG"
  grep -a "\[OP\]" /root/llm/test/$tag.log | tail -25 >> "$RUNLOG"
}

run e4a 0
run e4b 1
echo "E4_DONE rc=0" >> "$RUNLOG"
