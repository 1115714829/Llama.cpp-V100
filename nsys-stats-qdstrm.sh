#!/bin/bash
# nsys-stats-qdstrm.sh -- import the raw qdstrm stream directly (no finalization was run).
set -u
NSYS=/usr/local/cuda-12.4/bin/nsys
echo "=== import + kernel summary ==="
"$NSYS" stats --report cuda_gpu_kern_sum /tmp/prof-tgt.qdstrm 2>&1 | tail -45
echo NSYS_STATS_DONE
