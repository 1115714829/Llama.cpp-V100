#!/bin/bash
# BL1 stress wrapper: 1cat-vLLM standard service. The API key is extracted
# server-side from the unit file and never printed.
set -u
KEY=$(grep -oE 'sk-[A-Za-z0-9-]+' /root/llm/systemd/vllm-1cat.service | head -1)
python3 /root/llm/test/stress_client.py \
  --base-url http://127.0.0.1:8000 \
  --api-key "$KEY" \
  --model Qwen3.8-27B-FP8 \
  --prompt-tokens 235930 \
  --prompt-file /root/llm/test/bl-prompt90.txt \
  --gen 128 --reps 2 --tag ${TAG:-BL1}
