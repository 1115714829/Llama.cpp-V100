@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\mb-raw.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === size === > "%OUT%"
%SSHH% "wc -l /tmp/mb-mulmat.log" >> "%OUT%" 2>&1
echo === first 25 === >> "%OUT%"
%SSHH% "head -25 /tmp/mb-mulmat.log" >> "%OUT%" 2>&1
echo === last 30 === >> "%OUT%"
%SSHH% "tail -30 /tmp/mb-mulmat.log" >> "%OUT%" 2>&1
echo === any Q8_0 lines === >> "%OUT%"
%SSHH% "grep -ac Q8_0 /tmp/mb-mulmat.log" >> "%OUT%" 2>&1
%SSHH% "grep -a Q8_0 /tmp/mb-mulmat.log | head -12" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
