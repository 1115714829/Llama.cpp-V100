#!/bin/bash
# e7-verify-split.sh - build + gate + 256K verification of the FA split floor.
#
# GGML_CUDA_FA_SPLIT_FLOOR is now code-default 2 (E6 sweep: 64.9 vs 70.3 ms/round).
# This run confirms the default reproduces the win without any env.
set -u
RUNLOG=/root/llm/test/e7-run.log
: > "$RUNLOG"
echo "=== build ===" >> "$RUNLOG"
LOG=/root/llm/test/e7-build.log bash /tmp/p4-build.sh
grep -a -E "BUILD_RC|MANIFEST_DIFFS|LIBGGML_CUDA_MD5" /root/llm/test/e7-build.log >> "$RUNLOG"
if ! grep -q "BUILD_RC=0" /root/llm/test/e7-build.log; then
  echo "E7_DONE rc=1 build_failed" >> "$RUNLOG"
  exit 1
fi
echo "=== gate ===" >> "$RUNLOG"
SPEC=1 bash /root/llm/test/t1c-gate.sh >> "$RUNLOG" 2>&1
grep -a "GATE " /tmp/t1cg.log | tail -2 >> "$RUNLOG"
echo "=== 256K verify (no env, split floor default 2) ===" >> "$RUNLOG"
pkill -f 'llama-serve[r] --model' 2>/dev/null
sleep 4
tmux kill-session -t srv 2>/dev/null
tmux new-session -d -s srv "TAG=e7 SPEC=1 ROUND=1 UB=2048 bash /root/llm/test/t1c-run.sh"
code=000
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8082/health)
  [ "$code" = "200" ] && break
  sleep 5
done
echo "HEALTH code=$code" >> "$RUNLOG"
python3 /root/llm/test/stress_client.py --base-url http://127.0.0.1:8082 \
  --model Qwen3.8-27B-Q8_0-BL2 --prompt-tokens 235930 --prompt-file bl-prompt90.txt \
  --gen 320 --reps 1 --tag e7 --timeout 1500 >> "$RUNLOG" 2>&1
grep -a "draft acceptance" /root/llm/test/e7.log | tail -1 >> "$RUNLOG"
pkill -f 'llama-serve[r] --model' 2>/dev/null
sleep 4
echo "E7_DONE rc=0" >> "$RUNLOG"
