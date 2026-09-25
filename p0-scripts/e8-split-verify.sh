#!/bin/bash
# e8-split-verify.sh - same-window 3-arm check: code default 2 vs env=2 vs env=0.
#
# e7 (code default) showed 79.4 ms/round while e6f2 (env=2) showed 64.9. This run
# decides whether that gap is window noise or the default not taking effect.
# GGML_CUDA_OP_TIMING on the B arm gives the FA call time to confirm the split.
set -u
RUNLOG=/root/llm/test/e8-run.log
: > "$RUNLOG"
OPENV="GGML_CUDA_OP_TIMING=1 GGML_CUDA_DISABLE_GRAPHS=1 GGML_META_SUBGRAPH_CAPTURE=0 GGML_CUDA_NO_CONCURRENT=1"

run() {
  local tag=$1; shift
  local envs="$*"
  pkill -f 'llama-serve[r] --model' 2>/dev/null
  sleep 4
  tmux kill-session -t srv 2>/dev/null
  tmux new-session -d -s srv "TAG=$tag SPEC=1 ROUND=1 UB=2048 $envs bash /root/llm/test/t1c-run.sh"
  local code=000
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8082/health)
    [ "$code" = "200" ] && break
    sleep 5
  done
  echo "HEALTH $tag code=$code $envs" >> "$RUNLOG"
  if [ "$code" != "200" ]; then
    tail -4 /root/llm/test/$tag.log >> "$RUNLOG"
    return 1
  fi
  python3 /root/llm/test/stress_client.py --base-url http://127.0.0.1:8082 \
    --model Qwen3.8-27B-Q8_0-BL2 --prompt-tokens 235930 --prompt-file bl-prompt90.txt \
    --gen 320 --reps 1 --tag $tag --timeout 1500 >> "$RUNLOG" 2>&1
  grep -a "draft acceptance" /root/llm/test/$tag.log | tail -1 >> "$RUNLOG"
  grep -a "\[OP\]" /root/llm/test/$tag.log | tail -22 >> "$RUNLOG"
  pkill -f 'llama-serve[r] --model' 2>/dev/null
  sleep 4
}

run e8d "GGML_CUDA_FA_SPLIT_FLOOR=0"
run e8e "GGML_CUDA_FA_SPLIT_FLOOR=2 $OPENV"
run e8f "$OPENV"
echo "E8_DONE rc=0" >> "$RUNLOG"
