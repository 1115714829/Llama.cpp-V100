#!/bin/bash
set -u
pkill -f restore[.]sh
sleep 1
pkill -x cmake; pkill -x gmake; pkill -x nvcc; pkill -x cc1plus
sleep 2
cd /root/llm/test/v100-opt/llama.cpp || exit 1
echo '--- before ---'
ls src/llama-graph.cpp src/models/llama-graph.cpp 2>&1 | head -4
mv -f src/models/llama-graph.cpp src/models/llama-graph.h src/models/llama-memory-recurrent.cpp src/models/llama-memory-recurrent.h src/ 2>&1 | head -2
echo '--- after ---'
ls -la src/llama-graph.cpp src/llama-graph.h src/llama-memory-recurrent.cpp src/llama-memory-recurrent.h src/models/delta-net-base.cpp | awk '{print $5, $9}'
echo -n 'rs marker in source (want 0): '; grep -a -c GGML_RS_INDEX_WRITE src/llama-graph.cpp
echo -n 's_write in source (want 0): '; grep -a -c s_write src/llama-graph.cpp
echo -n 'device AR in allreduce.cu (want 0): '; grep -a -c ggml_cuda_ar_device_allreduce ggml/src/ggml-cuda/allreduce.cu
echo -n 'FA probe in fattn-common.cuh (want 1): '; grep -a -c GGML_CUDA_FA_SPLIT_FLOOR ggml/src/ggml-cuda/fattn-common.cuh
echo FIX_DONE