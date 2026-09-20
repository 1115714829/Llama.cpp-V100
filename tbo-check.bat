@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\tbo-check.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "cd /root/llm/test/v100-opt/llama.cpp; echo ===DIRS===; ls -d build* 2>/dev/null; echo ===FIND===; ls build/bin/test-backend-ops build-nccl/bin/test-backend-ops 2>&1 | head -4; echo ===BINLIST===; ls build/bin/ | head -30; echo ===TESTSFLAG===; grep -aE 'LLAMA_BUILD_TESTS' build/CMakeCache.txt build-nccl/CMakeCache.txt 2>/dev/null" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
