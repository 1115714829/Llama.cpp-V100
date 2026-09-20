@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\ppl-bf.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === ppl-bf tail 25 === > "%OUT%"
%SSHH% "tail -25 /tmp/ppl-bf.log" >> "%OUT%" 2>&1
echo === ppl-bf size === >> "%OUT%"
%SSHH% "wc -l /tmp/ppl-bf.log /tmp/ppl-nccl.log" >> "%OUT%" 2>&1
echo === ppl-nccl estimate === >> "%OUT%"
%SSHH% "grep -aE 'Final estimate' /tmp/ppl-nccl.log" >> "%OUT%" 2>&1
echo === finish-ppl tail === >> "%OUT%"
%SSHH% "tail -12 /tmp/finish-ppl.log" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
