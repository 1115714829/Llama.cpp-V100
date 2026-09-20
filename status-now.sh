#!/bin/bash
# Quick状态检查：有没有遗留进程/服务在跑，GPU 是否干净，产物是否在位。
set -uo pipefail
echo "=== date / load ==="
date; uptime
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== 相关 systemd 单元 ==="
for u in llama-server llama-server-test llama-server-c4 llama-server-l3-pristine llama-server-l3-c4 vllm-1cat llmscope new-api; do
  printf "%-26s %s\n" "$u" "$(systemctl is-active "$u" 2>/dev/null)"
done
echo "=== 遗留进程 ==="
pgrep -af "llama-server|llama-bench|mtp-sweep|spec-sweep|dflash|prompt-robust|l3-" | head -10 || echo "(none)"
echo "=== 库目录（变体）==="
ls -d /root/libdir-* 2>/dev/null
echo "=== 关键产物 ==="
ls -la /root/llm/test/v100-opt/llama.cpp/build/bin/llama-bench /root/llm/test/v100-opt/llama.cpp/build/bin/llama-server 2>/dev/null
echo "=== 源码是否只剩 C4+C5（应无 mmq-config-volta.cuh）==="
ls /root/llm/test/v100-opt/llama.cpp/ggml/src/ggml-cuda/mmq-config-volta.cuh 2>/dev/null || echo "mmq-config-volta.cuh 不存在 ✓"
grep -c "MMVQ_PARAMETERS_VOLTA" /root/llm/test/v100-opt/llama.cpp/ggml/src/ggml-cuda/mmvq.cu
grep -c "MMVQ_VOLTA_MAX_BATCH_SIZE_K" /root/llm/test/v100-opt/llama.cpp/ggml/src/ggml-cuda/mmvq.cuh
echo STATUS_DONE
