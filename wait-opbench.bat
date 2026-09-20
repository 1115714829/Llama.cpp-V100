@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\opbench-wait.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "for i in $(seq 1 40); do if grep -q OPBENCH_DONE /tmp/opbench.log 2>/dev/null; then break; fi; sleep 20; done; echo ===STATUS===; cat /tmp/opbench.log; echo ===FALINES===; wc -l < /tmp/opbench-fa.txt 2>/dev/null; echo ===MMLINES===; wc -l < /tmp/opbench-mm.txt 2>/dev/null" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
