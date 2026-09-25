#!/bin/bash
# e2-q8tile.sh - TILE q8_0 in-kernel dequant A/B (ABBA).
#
# Arm A = archived binary /root/libdir-gb-bak-20260925-1521 (TILE stages q8_0 -> f16).
# Arm B = /root/libdir-gb (libggml-cuda.so e497c968: TILE dequants q8_0 in-kernel).
# Only variable = the tile-kernel change. Gate sha already green (e2g).
set -u
LOG=/root/llm/test/e2.log
: > "$LOG"
OLDLIBS=/root/libdir-gb-bak-20260925-1521
NEWLIBS=/root/libdir-gb

run() {
  local tag=$1; local libs=$2; shift 2
  local envs="$*"
  pkill -f 'llama-serve[r] --model' 2>/dev/null
  sleep 4
  tmux kill-session -t srv 2>/dev/null
  tmux new-session -d -s srv "TAG=$tag SPEC=1 ROUND=1 UB=2048 LIBS=$libs $envs bash /root/llm/test/t1c-run.sh"
  local code=000
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8082/health)
    [ "$code" = "200" ] && break
    sleep 5
  done
  echo "HEALTH $tag code=$code libs=$libs" >> "$LOG"
  if [ "$code" != "200" ]; then
    tail -4 /root/llm/test/$tag.log >> "$LOG"
    return 1
  fi
  python3 /root/llm/test/stress_client.py --base-url http://127.0.0.1:8082 \
    --model Qwen3.8-27B-Q8_0-BL2 --prompt-tokens 235930 --prompt-file bl-prompt90.txt \
    --gen 320 --reps 1 --tag $tag --timeout 1500 >> "$LOG" 2>&1
  grep -a "draft acceptance" /root/llm/test/$tag.log | tail -1 >> "$LOG"
  pkill -f 'llama-serve[r] --model' 2>/dev/null
  sleep 4
}

echo "=== E2 ABBA: A=staged q8_0->f16, B=TILE q8_0 in-kernel ===" >> "$LOG"
run e2a1 "$OLDLIBS"
run e2b1 "$NEWLIBS"
run e2b2 "$NEWLIBS"
run e2a2 "$OLDLIBS"
echo "E2_DONE rc=0" >> "$LOG"
