#!/bin/bash
set -uo pipefail
cd /root/llm/test/v100-opt/llama.cpp
MODEL=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
echo "=== gguf metadata: context / rope / position ==="
head -c 4000000 "$MODEL" | strings -n 4 | grep -iE "context|rope|max_pos|position|seq_len|n_ctx|embd" | head -30
echo "=== llama-bench options: rope / flash / kv ==="
./build/bin/llama-bench --help 2>&1 | grep -iE "rope|flash|kv|cache|context|ctx" | head -30
echo CTX_DONE
