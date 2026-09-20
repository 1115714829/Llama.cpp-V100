@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\nccl-transport.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === NCCL version + init === > "%OUT%"
%SSHH% "grep -a 'NCCL INFO' /tmp/p60-env-debug-server.log" >> "%OUT%" 2>&1
echo === channels / transport summary === >> "%OUT%"
%SSHH% "grep -aE 'NCCL INFO (Channel|Connected|P2P|NVLS|via|Setting)' /tmp/p60-env-debug-server.log" >> "%OUT%" 2>&1
echo === env sweep progress === >> "%OUT%"
%SSHH% "cat /tmp/nccl-env-sweep.log" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
