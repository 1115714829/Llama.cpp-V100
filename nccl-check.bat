@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\nccl-check.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === ldconfig -p nccl === > "%OUT%"
%SSHH% "ldconfig -p" >> "%OUT%" 2>&1
echo === ls nccl libs === >> "%OUT%"
%SSHH% "ls -l /usr/lib64/libnccl.so* /usr/local/lib/libnccl.so* /usr/lib/powerpc64le-linux-gnu/libnccl.so*" >> "%OUT%" 2>&1
echo === ldd libggml-cuda === >> "%OUT%"
%SSHH% "ldd /root/libdir-rt/libggml-cuda.so.0.24.0" >> "%OUT%" 2>&1
echo === CMakeCache nccl === >> "%OUT%"
%SSHH% "grep -i nccl /root/llm/test/v100-opt/llama.cpp/build/CMakeCache.txt" >> "%OUT%" 2>&1
echo === GPU state === >> "%OUT%"
%SSHH% "nvidia-smi --query-gpu=index,memory.used --format=csv" >> "%OUT%" 2>&1
echo === running llama/vllm === >> "%OUT%"
%SSHH% "pgrep -af llama-server" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
