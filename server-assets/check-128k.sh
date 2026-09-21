#!/bin/bash
set -uo pipefail
cd /root/llm/test/v100-opt/llama.cpp
echo "=== 128k bench status (is it done? result?) ==="
if pgrep -f "llama-bench.*131072" >/dev/null 2>&1; then echo "STILL RUNNING"; else echo "DONE/exited"; fi
echo "--- 128k log tail ---"
tail -8 /tmp/bench128.log
echo "=== VMM env var / flag in source ==="
grep -rn "NO_VMM\|vmm_enabled\|GGML_VMM\|cudaVmm\|VMM" ggml/src/ggml-cuda/ggml-cuda.cu 2>/dev/null | head -12
echo "=== llama-bench: vmm / fit / kv options ==="
./build/bin/llama-bench --help 2>&1 | grep -iE "vmm|fit|offload|cache-type" | head -12
echo "128K_DONE"
