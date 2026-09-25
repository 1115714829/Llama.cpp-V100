#!/bin/bash
# e1-q8direct.sh - E1: decode verify reads q8_0 KV in-kernel (Split-D + q8-direct).
#
# Zero-code A/B. Arm A = stock (TILE + f16 staging = conversion tax). Arm B =
# Split-D decode kernel with in-kernel q8_0 dequant (no f16 mirror).
#   BENV enables: LLAMA_SM70_D256=1        (kernel admitted)
#                 LLAMA_SM70_79T_DECODE=1 (decode shapes admitted)
#                 LLAMA_SM70_Q8_DIRECT=1  (raw q8_0 block reads, no staging)
# Prefill keeps the 79T engine in both arms (query_len % 256 == 0 only).
#
# Pass criteria: (1) gate sha identical to bcda0092..., (2) TTFT no regression,
# (3) round cost down by the conversion tax (~9 ms), (4) spec-off pure step not
# slower (q=1 also routes to Split-D in arm B).
set -u
LOG=/root/llm/test/e1.log
: > "$LOG"
BENV="LLAMA_SM70_D256=1 LLAMA_SM70_79T_DECODE=1 LLAMA_SM70_Q8_DIRECT=1"

run() {
  local tag=$1; local spec=$2; shift 2
  local envs="$*"
  pkill -f 'llama-serve[r] --model' 2>/dev/null
  sleep 4
  tmux kill-session -t srv 2>/dev/null
  tmux new-session -d -s srv "TAG=$tag SPEC=$spec ROUND=1 UB=2048 $envs bash /root/llm/test/t1c-run.sh"
  local code=000
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8082/health)
    [ "$code" = "200" ] && break
    sleep 5
  done
  echo "HEALTH $tag code=$code" >> "$LOG"
  if [ "$code" != "200" ]; then
    tail -4 /root/llm/test/$tag.log >> "$LOG"
    return 1
  fi
  python3 /root/llm/test/stress_client.py --base-url http://127.0.0.1:8082 \
    --model Qwen3.8-27B-Q8_0-BL2 --prompt-tokens 235930 --prompt-file bl-prompt90.txt \
    --gen 320 --reps 1 --tag $tag --timeout 1500 >> "$LOG" 2>&1
  grep -a -E "draft acceptance|n_gen =" /root/llm/test/$tag.log | tail -2 >> "$LOG"
  grep -a -E "FAKD|ACCEPT: sm70|REJECT:|inject split|round timing" /root/llm/test/$tag.log | head -12 >> "$LOG"
  pkill -f 'llama-serve[r] --model' 2>/dev/null
  sleep 4
}

echo "=== E1 GATE (arm B env) ===" >> "$LOG"
SPEC=1 $BENV bash /root/llm/test/t1c-gate.sh >> "$LOG" 2>&1
grep -a -E "GATE |GATE_DONE" /root/llm/test/e1.log | tail -4 >> "$LOG"

echo "=== E1 256K spec-on ABBA (A=stock, B=q8-direct) ===" >> "$LOG"
run e1a1 1 ""
run e1b1 1 "$BENV GGML_CUDA_FA_KERNEL_DEBUG=1"
run e1b2 1 "$BENV"
run e1a2 1 ""

echo "=== E1 spec-off pure step (q=1 guard) ===" >> "$LOG"
run e1pa 0 ""
run e1pb 0 "$BENV"

echo "E1_DONE rc=0" >> "$LOG"
