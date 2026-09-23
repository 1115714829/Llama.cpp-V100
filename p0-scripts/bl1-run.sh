#!/bin/bash
# BL1 test launcher: 1cat-vLLM from LOCAL NVMe (/mnt/3.84t) for fast load.
# The box is diskless (NFS rootfs) and takes ~18 min to load 32 GB from NFS;
# the test NVMe loads in ~1-2 min. Model content verified identical to the
# /root/llm/models copies (config.json byte-same, 29G both) => numbers caliber
# unchanged, only load path changes. Paths under /mnt/3.84t also match the
# red-line "models only from /mnt/3.84t" rule.
# WITH_SPEC=1 -> include the delivered --speculative-config (standard line BL1);
# default -> spec-off (decomposition arm). Everything else mirrors
# /root/llm/systemd/vllm-1cat.service verbatim.
set -u
set -a; . /root/llm/systemd/1cat-runtime.env; set +a
export CUDA_VISIBLE_DEVICES=0,1,3,4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
KEY=$(grep -oE 'sk-[A-Za-z0-9-]+' /root/llm/systemd/vllm-1cat.service | head -1)
SPEC_ARGS=()
if [ "${WITH_SPEC:-}" = "1" ]; then
  SPEC_ARGS=(--speculative-config '{"method":"dflash","model":"/mnt/3.84t/Qwen3.8-27B-DFlash2-FP8","kv_cache_dtype":"auto","draft_sample_method":"probabilistic"}')
fi
exec /root/llm/ac922env/1cat-20260907/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /mnt/3.84t/Qwen3.8-27B-FP8 \
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
    "${SPEC_ARGS[@]}" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --chat-template /mnt/3.84t/Qwen3.8-27B-FP8/chat_template.jinja \
    --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key "$KEY"
