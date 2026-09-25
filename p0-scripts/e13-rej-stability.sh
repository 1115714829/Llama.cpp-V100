#!/bin/bash
# e13-rej-stability.sh - crash diagnosis for LLAMA_SPEC_REJ=1 (IMA in rep2 of e12).
# 4 reps, same config, to see whether the crash is REJ-caused or data-dependent.
set -u
RUNLOG=/root/llm/test/e13-run.log
: > "$RUNLOG"
tmux kill-session -t srv 2>/dev/null
sleep 3
tmux new-session -d -s srv "TAG=e13 SPEC=1 ROUND=1 UB=2048 LLAMA_SPEC_REJ=1 GGML_CUDA_FA_KERNEL_DEBUG=1 bash /root/llm/test/t1c-run.sh"
code=000
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8082/health)
  [ "$code" = "200" ] && break
  sleep 5
done
echo "HEALTH code=$code" >> "$RUNLOG"
if [ "$code" != "200" ]; then
  tail -6 /root/llm/test/e13.log >> "$RUNLOG"
  echo "E13_DONE rc=1 health" >> "$RUNLOG"
  exit 1
fi
python3 /root/llm/test/stress_client.py --base-url http://127.0.0.1:8082 \
  --model Qwen3.8-27B-Q8_0-BL2 --prompt-tokens 235930 --prompt-file bl-prompt90.txt \
  --gen 320 --reps 4 --tag e13 --timeout 1500 >> "$RUNLOG" 2>&1
grep -a "draft acceptance" /root/llm/test/e13.log | tail -4 >> "$RUNLOG"
grep -a -c "CUDA error" /root/llm/test/e13.log >> "$RUNLOG" 2>&1
tmux kill-session -t srv 2>/dev/null
sleep 5
echo "E13_DONE rc=0" >> "$RUNLOG"
