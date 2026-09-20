#!/bin/bash
# Read-only state gather on AC922 (V100). Touches nothing.
set -uo pipefail
echo "=== date ==="
date
echo ""
echo "=== GPU (index,name,mem_used,mem_total,util) ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""
echo "=== systemd units ==="
for u in llama-server llama-server-test llama-server-c4 vllm-1cat; do
  printf "%-20s %s\n" "$u" "$(systemctl is-active "$u" 2>/dev/null)"
done
echo ""
echo "=== journalctl llama-server-c4 : timing lines (tail 40) ==="
journalctl -u llama-server-c4 --no-pager -o cat 2>/dev/null | grep -E "prompt eval time|^eval time|draft acceptance|total time|prompt processing" | tail -40
echo ""
echo "=== journalctl llama-server-test : timing lines (tail 40) ==="
journalctl -u llama-server-test --no-pager -o cat 2>/dev/null | grep -E "prompt eval time|^eval time|draft acceptance|total time|prompt processing" | tail -40
echo ""
echo "=== /tmp 256k artifacts ==="
ls -la /tmp/prompt256k.txt /tmp/req256k.json /tmp/resp256k.json 2>/dev/null
ls -la /tmp/*256k* 2>/dev/null
echo ""
echo "=== binaries ==="
ls -la /root/llm/test/v100-opt/llama.cpp/build/bin/llama-server 2>/dev/null
ls -la /root/llm/test/v100-opt/llama.cpp/build/bin/llama-bench 2>/dev/null
ls -la /root/llm/llama.cpp/bin/llama-server 2>/dev/null
echo ""
echo "=== C4 source state (VOLTA refs in mmvq.cu) ==="
grep -c VOLTA /root/llm/test/v100-opt/llama.cpp/ggml/src/ggml-cuda/mmvq.cu 2>/dev/null
echo "=== C4 backup present? ==="
ls -la /root/mmvq-c4-backup.cu /root/mmvq-orig.cu 2>/dev/null
echo ""
echo "=== systemd unit files for our test servers ==="
ls -la /etc/systemd/system/llama-server-test.service /etc/systemd/system/llama-server-c4.service 2>/dev/null
echo ""
echo "=== free -g ==="
free -g
echo GATHER_DONE
