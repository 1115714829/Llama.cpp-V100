@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\state4.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === clock === > "%OUT%"
%SSHH% "date" >> "%OUT%" 2>&1
echo === final-bf log === >> "%OUT%"
%SSHH% "cat /tmp/p60-final-bf-tp3.log" >> "%OUT%" 2>&1
echo === server log tail === >> "%OUT%"
%SSHH% "tail -6 /tmp/p60-final-bf-tp3-server.log" >> "%OUT%" 2>&1
echo === gpu === >> "%OUT%"
%SSHH% "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader" >> "%OUT%" 2>&1
echo === mem === >> "%OUT%"
%SSHH% "free -g" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
