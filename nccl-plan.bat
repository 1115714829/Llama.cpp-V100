@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\nccl-plan.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
set PKG=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime
echo === nccl ldd deps === > "%OUT%"
%SSHH% "ldd %PKG%/lib/libnccl.so.2" >> "%OUT%" 2>&1
echo === does plain libnccl.so exist === >> "%OUT%"
%SSHH% "ls -l %PKG%/lib/" >> "%OUT%" 2>&1
echo === build arch + nccl cache === >> "%OUT%"
%SSHH% "grep -i 'CUDA_ARCHITECTURES\|NCCL\|CMAKE_BUILD_TYPE\|CMAKE_CUDA_FLAGS\|GGML_CUDA_GRAPHS\|GGML_CUDA_PEER' /root/llm/test/v100-opt/llama.cpp/build/CMakeCache.txt" >> "%OUT%" 2>&1
echo === free / disk === >> "%OUT%"
%SSHH% "free -g" >> "%OUT%" 2>&1
%SSHH% "df -h /root" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
