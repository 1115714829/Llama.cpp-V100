@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\ncu-out2.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "echo ===MARK===; grep -c NCU_TGT_DONE /tmp/ncu-tgt.log 2>/dev/null; echo ===KEY===; grep -aE 'ncu version|NCU_TGT_DONE|health after|tokens per second|CUDA error|Error|error|Duration|GATED|FLASH|MUL_MAT|====|WARNING|==PROF==|per-second' /tmp/ncu-tgt.log | head -70; echo ===TAIL===; tail -20 /tmp/ncu-tgt.log; echo ===REPFILE===; ls -la /tmp/ncu-tgt* 2>/dev/null" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
