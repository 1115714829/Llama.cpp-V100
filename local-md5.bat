@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\local-md5-13.txt
cd /d F:\vllm+llama.cpp\llama.cpp
> "%OUT%" echo === local md5 of the 13 modified files ===
for %%F in (common\common.cpp common\sampling.cpp common\speculative.cpp common\speculative.h ggml\src\ggml-cuda\mmvq.cu ggml\src\ggml-cuda\mmvq.cuh src\llama-context.cpp src\llama-context.h src\llama-ext.h src\llama-model.cpp src\llama-model.h src\models\dflash.cpp tools\server\server-context.cpp) do (certutil -hashfile "%%F" MD5 >> "%OUT%" 2>&1)
echo DONE >> "%OUT%"
