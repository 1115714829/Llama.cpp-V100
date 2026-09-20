@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\cleanup-check.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "pkill -f 'NCU_TGT_DON[E]' >/dev/null 2>&1; sleep 1; echo ===LEFT===; pgrep -af 'NCU_TGT_DON[E]' | head -3; echo ===TBO===; ls -la /root/llm/test/v100-opt/llama.cpp/build/bin/test-backend-ops 2>/dev/null || echo NO_TBO; echo ===NCUVER===; /usr/local/cuda-12.4/bin/ncu --version 2>&1 | head -3" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
