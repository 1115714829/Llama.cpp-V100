@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\vllm1cat-env.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === vllm-1cat unit (READ ONLY) === > "%OUT%"
%SSHH% "systemctl cat vllm-1cat" >> "%OUT%" 2>&1
echo === any NCCL/NVLink related env on the box === >> "%OUT%"
%SSHH% "grep -rIl 'NCCL_\|CUDA_DEVICE_ORDER' /root/llm/ /etc/systemd/system/ 2>/dev/null | head -20" >> "%OUT%" 2>&1
echo === 1cat serve script env === >> "%OUT%"
%SSHH% "ls /root/llm/ 2>&1 | head -30" >> "%OUT%" 2>&1
%SSHH% "grep -aE 'NCCL|CUDA_DEVICE|CUDA_VISIBLE|--tensor-parallel|[^-]--kv-cache|--max-model-len|gpu-memory-util' /root/p32-vllm1cat-bench.sh 2>/dev/null | head -20" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
