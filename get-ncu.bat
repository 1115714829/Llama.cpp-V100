@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\ncu-out.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do if grep -q NCU_TGT_DONE /tmp/ncu-tgt.log 2>/dev/null; then break; fi; sleep 25; done; echo ===LOG===; grep -aE 'ncu version|health after|tokens per second|CUDA error|error|Error|ncu-rep|===|Duration|GATED|FLASH|MUL_MAT|MUL_MAT_VEC|top|^ *[0-9]' /tmp/ncu-tgt.log | head -60; echo ===FULLTAIL===; tail -25 /tmp/ncu-tgt.log" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
