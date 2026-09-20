@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\mb-out.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do if grep -q MICROBENCH_DONE /tmp/microbench.log 2>/dev/null; then break; fi; sleep 20; done; cat /tmp/microbench.log" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
