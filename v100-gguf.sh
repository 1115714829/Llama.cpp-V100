#!/bin/bash
# Inspect the model's gguf metadata: arch, layers, and MTP/NextN module presence.
set -uo pipefail
cd /root/llm/test/v100-opt/llama.cpp
MODEL=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
echo "=== available tools (gguf/inspect) ==="
ls build/bin/ | grep -iE "gguf|imatrix|quantize|perplexity|bench|cli" | head -20
echo "=== llama-gguf metadata (arch + MTP/NextN keys) ==="
if [ -x build/bin/llama-gguf ]; then
  build/bin/llama-gguf --show "$MODEL" 2>&1 | grep -iE "arch|layer|nextn|mtp|predict|attention|head|embd|block" | head -60
else
  echo "no llama-gguf tool; trying llama-cli model print"
  CUDA_VISIBLE_DEVICES=2 build/bin/llama-cli -m "$MODEL" -p "x" -n 0 --verbose 2>&1 | grep -iE "arch|layer|nextn|mtp|head|embd|context" | head -40
fi
echo GGUF_DONE
