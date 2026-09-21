#!/bin/bash
# Health check for the background 256k A/B: is it actually progressing?
set -uo pipefail
L=/tmp/big-ab-256k.log
echo "now: $(date -Is)"
echo "log size: $(wc -c < "$L") bytes, $(wc -l < "$L") lines"
echo "--- markers ---"
grep -E "^(===|--- )" "$L" | tail -6
echo "--- llama-bench process ---"
ps -eo pid,etime,pcpu,comm --sort=-pcpu | grep -E "llama-bench" | grep -v grep || echo "(no llama-bench process)"
echo "--- GPU util (0,1) ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "--- python/tmp files touched recently ---"
ls -la /tmp/big-ab-256k.log
echo "--- any curl/timeout processes? ---"
ps -eo pid,etime,comm | grep -E "curl|timeout" | grep -v grep || echo "(none)"
echo HEALTH_DONE
