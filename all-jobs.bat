@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\all-jobs.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === jobs running === > "%OUT%"
%SSHH% "pgrep -af 'nccl-env-sweep|final-nccl|nccl-knobs2|finish-ppl|llama-server|llama-perplexity'" >> "%OUT%" 2>&1
echo === FINAL log === >> "%OUT%"
%SSHH% "cat /tmp/final-nccl.log" >> "%OUT%" 2>&1
echo === ENV sweep medians === >> "%OUT%"
%SSHH% "grep -aE '^##### ENV ARM|MEDIAN_TG=' /tmp/nccl-env-sweep.log" >> "%OUT%" 2>&1
echo === KNOBS2 log === >> "%OUT%"
%SSHH% "cat /tmp/nccl-knobs2.log" >> "%OUT%" 2>&1
echo === PPL log === >> "%OUT%"
%SSHH% "cat /tmp/finish-ppl.log" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
