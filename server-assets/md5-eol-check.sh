#!/bin/bash
# Compare the server copy of each modified file against itself with CRLF line endings,
# so we can tell "content differs" from "line endings differ".
cd /root/llm/test/v100-opt/llama.cpp || exit 1
for f in common/common.cpp common/sampling.cpp common/speculative.cpp common/speculative.h \
         ggml/src/ggml-cuda/mmvq.cu ggml/src/ggml-cuda/mmvq.cuh \
         src/llama-context.cpp src/llama-context.h src/llama-ext.h src/llama-model.cpp \
         src/llama-model.h src/models/dflash.cpp tools/server/server-context.cpp; do
  a=$(md5sum "$f" | awk '{print $1}')
  b=$(sed 's/$/\r/' "$f" | md5sum | awk '{print $1}')
  printf '%-45s server_lf=%s  server_as_crlf=%s\n' "$f" "$a" "$b"
done
echo "--- line-ending sample (first CR byte if any) ---"
file common/speculative.cpp src/llama-context.cpp ggml/src/ggml-cuda/mmvq.cu
echo EOLCHECK_DONE
