@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\opbench-gdn-local.txt
ssh -o BatchMode=yes root@192.168.50.235 "cat /tmp/opbench-gdn.txt; echo ===FA_PROGRESS===; wc -l < /tmp/opbench-fa.txt 2>/dev/null; tail -2 /tmp/opbench-fa.txt 2>/dev/null; echo ===MAINLOG===; cat /tmp/opbench.log" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
