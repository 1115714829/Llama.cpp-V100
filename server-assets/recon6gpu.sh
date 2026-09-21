#!/bin/bash
# Read-only reconnaissance after GPU reservation was lifted (2026-09-20).
set -uo pipefail

echo "=== [1] /root/llm/models/ ==="
ls -la /root/llm/models/
echo ""
echo "=== [2] TurboFCFusion dir ==="
ls -la /root/llm/models/Qwen3.8-27B-TurboFCFusion-gguf/
echo ""
echo "=== [2b] any README / config / json in that dir ==="
find /root/llm/models/Qwen3.8-27B-TurboFCFusion-gguf/ -maxdepth 1 -type f \( -name "*.md" -o -name "*.json" -o -name "*.txt" \) 2>/dev/null
echo ""
echo "=== [3] original production unit (DO NOT MODIFY): /root/llm/systemd/llama-server.service ==="
grep -vE "^#|^$" /root/llm/systemd/llama-server.service 2>/dev/null
echo ""
echo "=== [4] nvidia-smi compute apps ==="
nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv 2>/dev/null
echo ""
echo "=== [5] nvidia-smi memory per GPU ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""
echo "=== [6] processes: vllm / llmscope / llama (top RSS) ==="
ps -eo pid,ppid,user,etime,rss,comm,args --sort=-rss 2>/dev/null | grep -iE "vllm|llmscope|llama|EngineCore" | grep -v grep | cut -c1-200
echo ""
echo "=== [7] systemd units of interest ==="
systemctl list-units --type=service --all --no-pager 2>/dev/null | grep -iE "vllm|llama|llmscope" || echo "(none)"
echo ""
echo "=== [8] llama.cpp bin dir (orig install) ==="
ls /root/llm/llama.cpp/bin/ 2>/dev/null
echo ""
echo "=== [9] topology ==="
nvidia-smi topo -m 2>/dev/null | head -14
echo ""
echo "=== [10] free -g ==="
free -g
echo ""
echo "=== [11] /etc/systemd/system custom units ==="
systemctl list-unit-files --no-pager 2>/dev/null | grep -iE "vllm|llama" || echo "(none)"
echo RECON_DONE
