@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\collect2.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40; do if grep -q COLLECT2_DONE /root/collect2.out 2>/dev/null; then break; fi; sleep 30; done; cat /root/collect2.out" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
