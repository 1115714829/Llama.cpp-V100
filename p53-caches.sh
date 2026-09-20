#!/bin/bash
# Test the diagnosis: if the GPU "Used" memory is HBM2-backed page cache, dropping caches frees it.
set -uo pipefail
echo "=== BEFORE ==="
free -g | head -2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo ""
echo "=== sync + drop page cache (safe: cache simply refills) ==="
sync
echo 3 > /proc/sys/vm/drop_caches
sleep 6

echo "=== AFTER ==="
free -g | head -2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo ""
echo "=== holders re-check ==="
fuser /dev/nvidia3 2>/dev/null || true
echo P53_DONE
