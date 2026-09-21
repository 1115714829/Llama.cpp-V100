#!/bin/bash
# Classify the residue: driver-reserved vs a real allocation vs HBM2-mapped-as-system-RAM.
set -uo pipefail
echo "=== GPU 3 memory detail ==="
nvidia-smi -q -i 3 2>/dev/null | grep -A 12 -e "FB Memory Usage" -e "BAR1 Memory Usage" | head -40

echo ""
echo "=== GPU 0 memory detail (clean reference) ==="
nvidia-smi -q -i 0 2>/dev/null | grep -A 8 "FB Memory Usage" | head -14

echo ""
echo "=== system RAM (PPC64LE maps HBM2 into system memory here) ==="
free -g
echo ""
echo "=== driver version / persistence mode ==="
nvidia-smi --query-gpu=index,persistence_mode,memory.used,memory.total --format=csv
echo ""
echo "=== note: any 'reserved' accounting shown as used? full FB block for GPU 1 ==="
nvidia-smi -q -i 1 2>/dev/null | sed -n '/FB Memory Usage/,/^$/p' | head -12
echo P52_DONE
