@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\final-progress.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do if grep -q FINAL_NCCL_DONE /tmp/final-nccl.log 2>/dev/null; then break; fi; sleep 30; done; echo ===FINAL===; cat /tmp/final-nccl.log; echo ===ENVS===; grep -aE '^##### ENV ARM|MEDIAN_TG=' /tmp/nccl-env-sweep.log; echo ===KNOBS===; grep -aE '^##### KNOB ARM|MEDIAN_TG=' /tmp/nccl-knobs2.log 2>/dev/null" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
