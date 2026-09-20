@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\local-patch-check.txt
cd /d F:\vllm+llama.cpp\llama.cpp
echo === diff of the two patched files === > "%OUT%"
git diff -- ggml/src/ggml-cuda/mmvq.cuh ggml/src/ggml-cuda/mmvq.cu >> "%OUT%" 2>&1
echo === status count === >> "%OUT%"
git status --porcelain >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
