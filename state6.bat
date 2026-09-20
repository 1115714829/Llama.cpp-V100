@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\state6.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === clock === > "%OUT%"
%SSHH% "date" >> "%OUT%" 2>&1
echo === procs === >> "%OUT%"
%SSHH% "pgrep -af 'q8mmq-ab|fina[l]-nccl|pp[l]-after|collect2|llama-serve[r]|cmake|ninja|cc1plus|nvcc' | head -12" >> "%OUT%" 2>&1
echo === q8 log === >> "%OUT%"
%SSHH% "cat /tmp/q8mmq-ab.log" >> "%OUT%" 2>&1
echo === collect2.out === >> "%OUT%"
%SSHH% "cat /root/collect2.out" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
