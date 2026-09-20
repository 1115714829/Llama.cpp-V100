#!/bin/bash
set -uo pipefail
cd /root/llm/test/v100-opt/llama.cpp
MODEL=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
echo "=== llama-gguf tool present? ==="
ls -la build/bin/llama-gguf 2>&1
echo "=== models dir (look for MTP/nextn draft gguf) ==="
ls -la /root/llm/models/Qwen3.8-27B-GGUF/ 2>&1
echo "=== all model folders ==="
ls /root/llm/models/ 2>&1
echo "=== any mtp/nextn gguf anywhere ==="
find /root/llm/models/ -iname "*mtp*" -o -iname "*nextn*" 2>/dev/null | head
echo "=== llama-gguf help (syntax) ==="
build/bin/llama-gguf --help 2>&1 | head -25
echo MTP_DONE
