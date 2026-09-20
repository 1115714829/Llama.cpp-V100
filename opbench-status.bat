@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\opbench-status.txt
ssh -o BatchMode=yes root@192.168.50.235 "echo ===LOG===; cat /tmp/opbench.log; echo ===PROC===; pgrep -af 'test-backend-op[s]' | head -2; echo ===FA===; wc -l < /tmp/opbench-fa.txt 2>/dev/null; tail -1 /tmp/opbench-fa.txt 2>/dev/null | cut -c1-140" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
