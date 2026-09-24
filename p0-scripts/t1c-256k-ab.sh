#!/bin/bash
# T1-C 256K A/B: arm k1 = 79T engine ON (LLAMA_SM70_79T=1), arm k0 = fused sm70 path.
# Same source, same prompt file, same UB, serial (GPU exclusive). Marks: AB_ARM_*, AB_DONE.
set -u
LOG=/tmp/ab.log
: > "$LOG"
PT=${PT:-235930}
PF=/root/llm/test/bl-prompt90.txt
GEN=${GEN:-4}
REPS=${REPS:-1}

run_arm() {
  local tag=$1; shift
  local extra="$*"
  echo "=== AB_ARM_START $tag extra=[$extra] $(date +%H:%M:%S) ===" >> "$LOG"
  pkill -f 'llama-serve[r] --model' 2>/dev/null
  sleep 4
  tmux kill-session -t srv 2>/dev/null
  rm -f /root/llm/test/$tag.log /root/llm/test/$tag-stress.log
  tmux new-session -d -s srv "TAG=$tag UB=2048 $extra bash /root/llm/test/t1c-run.sh"
  local code=000
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8082/health)
    if [ "$code" = "200" ]; then break; fi
    sleep 5
  done
  echo "AB_HEALTH $tag code=$code tries=$i" >> "$LOG"
  if [ "$code" != "200" ]; then
    tail -5 /root/llm/test/$tag.log >> "$LOG"
    echo "AB_ARM_END $tag FAILED" >> "$LOG"
    return 1
  fi
  python3 /root/llm/test/stress_client.py --base-url http://127.0.0.1:8082 \
    --model Qwen3.8-27B-Q8_0-BL2 --prompt-tokens "$PT" --prompt-file "$PF" \
    --gen "$GEN" --reps "$REPS" --tag "$tag" --timeout 1500 \
    >> /root/llm/test/$tag-stress.log 2>&1
  grep -aE 'STRESS tag|STRESS-BEGIN' /root/llm/test/$tag-stress.log | tail -2 >> "$LOG"
  grep -aoE 'total=[0-9.]+s calls=[0-9]+' /root/llm/test/$tag.log | tail -1 >> "$LOG"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -4 >> "$LOG"
  pkill -f 'llama-serve[r] --model' 2>/dev/null
  sleep 4
  echo "AB_ARM_END $tag $(date +%H:%M:%S)" >> "$LOG"
}

for a in ${ARMS:-k1 k0}; do
  if [ "$a" = "k1" ]; then run_arm k1 ""; else run_arm k0 "NO79T=1"; fi
done
echo "AB_DONE $(date +%H:%M:%S)" >> "$LOG"
