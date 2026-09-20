@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\ab-now.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === summary === > "%OUT%"
%SSHH% "cat /tmp/nccl-ab-summary.log" >> "%OUT%" 2>&1
echo === ab log === >> "%OUT%"
%SSHH% "cat /tmp/nccl-ab.log" >> "%OUT%" 2>&1
echo === running === >> "%OUT%"
%SSHH% "pgrep -af 'llama-server|nccl-ab'" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
