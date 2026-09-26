#!/bin/bash
# bl1nospec.sh - BL1 pure-decode baseline: 1cat-vLLM without speculative config.
#
# Copy of the production unit's ExecStart minus --speculative-config (DFlash2),
# minus --api-key, on port 8001. Never touches the production unit or port 8000.
set -u
RUNLOG=/root/llm/test/bl1nospec-run.log
: > "$RUNLOG"

cat > /tmp/bl1-start.sh <<'EOF'
#!/bin/bash
source /root/llm/systemd/1cat-runtime.env
export CUDA_VISIBLE_DEVICES=0,1,3,4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /root/llm/ac922env/1cat-20260907/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /root/llm/models/Qwen3.8-27B-FP8 \
    --served-model-name Qwen3.8-27B-FP8 \
    --trust-remote-code \
    --attention-backend FLASH_ATTN_V100 \
    --tensor-parallel-size 4 \
    --dtype half \
    --kv-cache-dtype fp8_e5m2 \
    --gpu-memory-utilization 0.905 \
    --max-model-len 262144 \
    --max-num-seqs 1 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --max-num-batched-tokens 2048 \
    --limit-mm-per-prompt '{"image":999,"video":0}' \
    --mm-processor-kwargs '{"max_pixels":2073680}' \
    --mm-processor-cache-gb 0 \
    --chat-template /root/llm/models/Qwen3.8-27B-FP8/chat_template.jinja \
    --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
    --host 127.0.0.1 \
    --port 8001
EOF
chmod +x /tmp/bl1-start.sh

tmux kill-session -t bl1 2>/dev/null
tmux new-session -d -s bl1 "bash /tmp/bl1-start.sh > /root/llm/test/bl1nospec.log 2>&1"

code=000
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8001/health)
  [ "$code" = "200" ] && break
  sleep 5
done
echo "HEALTH code=$code" >> "$RUNLOG"
if [ "$code" != "200" ]; then
  tail -6 /root/llm/test/bl1nospec.log >> "$RUNLOG"
  tmux kill-session -t bl1 2>/dev/null
  echo "BL1NOSPEC_DONE rc=1 health" >> "$RUNLOG"
  exit 1
fi
python3 /root/llm/test/stress_client.py --base-url http://127.0.0.1:8001 \
  --model Qwen3.8-27B-FP8 --prompt-tokens 235930 --prompt-file bl-prompt90.txt \
  --gen 320 --reps 2 --tag bl1ns --timeout 1500 >> "$RUNLOG" 2>&1
tmux kill-session -t bl1 2>/dev/null
sleep 6
echo "BL1NOSPEC_DONE rc=0" >> "$RUNLOG"
