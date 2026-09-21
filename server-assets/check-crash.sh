#!/bin/bash
set -uo pipefail
echo "=== full stress-orig.out (CUDA error / crash details) ==="
grep -iE "cuda|error|abort|assert|fail|memory|OOM|out of|kernel|illegal|misaligned|smem|shared" /tmp/stress-orig.out | head -40
echo "=== first 30 lines (the load + context) ==="
head -30 /tmp/stress-orig.out
echo "=== last 25 lines (the crash) ==="
tail -25 /tmp/stress-orig.out
echo CRASH_DONE
