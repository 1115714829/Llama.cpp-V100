@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\state2.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === queue === > "%OUT%"
%SSHH% "pgrep -af 'nc[c]l-env-sweep|fina[l]-nccl|pp[l]-after-final|llama-perplex[ity]|llama-serve[r]' | head -8" >> "%OUT%" 2>&1
echo === sweep arms === >> "%OUT%"
%SSHH% "grep -aE '^##### ENV ARM|MEDIAN_TG=|SWEEP_DONE' /tmp/nccl-env-sweep.log" >> "%OUT%" 2>&1
echo === final log === >> "%OUT%"
%SSHH% "cat /tmp/final-nccl.log" >> "%OUT%" 2>&1
echo === ppl log === >> "%OUT%"
%SSHH% "cat /tmp/finish-ppl.log" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
