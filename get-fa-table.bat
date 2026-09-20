@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\fa-table.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes fa-summarize.sh root@192.168.50.235:/root/ >nul
%SSHH% "bash /root/fa-summarize.sh" > "%OUT%" 2>&1
%SSHH% "echo ===MMPROC===; cat /tmp/opbench.log" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
