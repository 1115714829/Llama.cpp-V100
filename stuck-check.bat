@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\stuck-check.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === time === > "%OUT%"
%SSHH% "date" >> "%OUT%" 2>&1
echo === arm logs by mtime === >> "%OUT%"
%SSHH% "ls -lt /tmp/p60-env-1ch.log /tmp/p60-env-1ch-server.log /tmp/p60-env-ring1ch.log /tmp/final-nccl.log" >> "%OUT%" 2>&1
echo === env-1ch server tail === >> "%OUT%"
%SSHH% "tail -5 /tmp/p60-env-1ch-server.log" >> "%OUT%" 2>&1
echo === env-1ch summary === >> "%OUT%"
%SSHH% "cat /tmp/p60-env-1ch.log" >> "%OUT%" 2>&1
echo === sweep log tail === >> "%OUT%"
%SSHH% "tail -14 /tmp/nccl-env-sweep.log" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
