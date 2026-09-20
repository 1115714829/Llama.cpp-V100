@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\devd-crash.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === devd-tensor server log (full) === > "%OUT%"
%SSHH% "cat /tmp/p60-devd-tensor-server.log" >> "%OUT%" 2>&1
echo === env sweep log === >> "%OUT%"
%SSHH% "cat /tmp/nccl-env-sweep.log" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
