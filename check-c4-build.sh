#!/bin/bash
set -uo pipefail
C4BUILD=/root/llm/test/v100-opt/llama.cpp/build/bin
echo "=== C4 build bin (llama-server present?) ==="
ls -la "$C4BUILD"/llama-server "$C4BUILD"/llama-bench 2>/dev/null
echo "=== C4 build: which mmvq.cu is in source (C4 or original)? ==="
grep -c "MMVQ_PARAMETERS_VOLTA" /root/llm/test/v100-opt/llama.cpp/ggml/src/ggml-cuda/mmvq.cu 2>/dev/null
echo "=== C4 backup (mmvq-c4-backup.cu) present? ==="
ls -la /root/mmvq-c4-backup.cu 2>/dev/null
grep -c "MMVQ_PARAMETERS_VOLTA" /root/mmvq-c4-backup.cu 2>/dev/null
echo "=== current source mmvq.cu: VOLTA count (C4=5+, orig=0) ==="
grep -n "MMVQ_PARAMETERS_VOLTA" /root/llm/test/v100-opt/llama.cpp/ggml/src/ggml-cuda/mmvq.cu 2>/dev/null | head -3
echo C4CHECK_DONE
