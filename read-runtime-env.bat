@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\1cat-runtime-env.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === /root/llm/systemd/1cat-runtime.env (READ ONLY) === > "%OUT%"
%SSHH% "cat /root/llm/systemd/1cat-runtime.env" >> "%OUT%" 2>&1
echo === 1cat-gpu-init service (what it sets up) === >> "%OUT%"
%SSHH% "systemctl cat 1cat-gpu-init 2>&1 | head -30" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
