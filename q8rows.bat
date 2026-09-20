@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\q80rows.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === q8_0 MUL_MAT rows (lowercase in the log) === > "%OUT%"
%SSHH% "grep -a -A1 'type_a=q8_0' /tmp/mb-mulmat.log" >> "%OUT%" 2>&1
echo === q4_0 (also dp4a legacy, MMVQ path) === >> "%OUT%"
%SSHH% "grep -a -A1 'type_a=q4_0' /tmp/mb-mulmat.log" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
