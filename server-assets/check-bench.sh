#!/bin/bash
set -uo pipefail
cd /root/llm/test/v100-opt/llama.cpp
echo "=== llama-bench version ==="
build/bin/llama-bench --version 2>&1 | head -3
echo "=== llama-bench help (multi-gpu / context / batch options) ==="
build/bin/llama-bench --help 2>&1 | grep -iE "tensor.split|n.gpu|ngl|context|batch|model|prompt|n-prompt|-p | -n " | head -40
echo "=== GPU 2/5 free memory ==="
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader -i 2 -i 5
echo BENCH_DONE
