#!/bin/bash
# (a) Is a profiler actually available on this box? (the C4 premise cites ncu profiling)
# (b) Record what profiling tools exist, so future claims have provenance.
set -uo pipefail
echo "=== which ncu / nsys / nvprof ==="
command -v ncu nsys nvprof 2>&1 || echo "(none on PATH)"
echo ""
echo "=== CUDA bin dir ==="
ls /usr/local/cuda-12.4/bin/ 2>/dev/null | head -40
echo ""
echo "=== any ncu/nsys binary anywhere under /usr/local /opt ==="
find /usr/local /opt -maxdepth 5 -name "ncu" -o -maxdepth 5 -name "nsys" 2>/dev/null | head -10
echo ""
echo "=== cupti / profiler libs ==="
ls /usr/local/cuda-12.4/extras/CUPTI/lib64/ 2>/dev/null | head -8 || echo "(no CUPTI)"
echo ""
echo "=== L0/L1 timing methodology available instead: llama-bench -b/-ub (already used) ==="
echo "PROFILER_CHECK_DONE"
