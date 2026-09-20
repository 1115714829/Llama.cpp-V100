@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\prof-files.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === /tmp prof-tgt* === > "%OUT%"
%SSHH% "ls -l /tmp/prof-tgt* /tmp/*.nsys-rep" >> "%OUT%" 2>&1
echo === any nsys-rep anywhere recent === >> "%OUT%"
%SSHH% "find /root /tmp -maxdepth 3 -name '*.nsys-rep' -newermt '-40 minutes'" >> "%OUT%" 2>&1
echo === server log tail === >> "%OUT%"
%SSHH% "tail -6 /tmp/prof-tgt-server.log" >> "%OUT%" 2>&1
echo === nsys still running? === >> "%OUT%"
%SSHH% "pgrep -af 'nsys|llama-server'" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
