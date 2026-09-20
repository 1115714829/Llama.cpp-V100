#!/bin/bash
# Clean up leftover VRAM / orphaned test processes before measuring.
set -uo pipefail
echo "=== BEFORE: compute apps ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null
echo "=== BEFORE: per-GPU memory ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo ""
echo "=== leftover test processes? ==="
ps -eo pid,ppid,stat,etime,cmd 2>/dev/null | grep -e llama-server -e llama-bench -e llama-cli | grep -v grep || echo "(none)"

echo ""
echo "=== terminating leftovers ==="
pkill -f llama-server 2>/dev/null || true
pkill -f llama-bench  2>/dev/null || true
sleep 5
pkill -9 -f llama-server 2>/dev/null || true
pkill -9 -f llama-bench  2>/dev/null || true
sleep 8

echo "=== AFTER: compute apps ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null
echo "=== AFTER: per-GPU memory ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== any defunct/zombie? ==="
ps -eo pid,stat,cmd 2>/dev/null | grep -e "<defunct>" | grep -v grep || echo "(no zombies)"
echo P50_DONE
