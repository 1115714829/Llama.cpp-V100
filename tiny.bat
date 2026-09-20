@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\tiny.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "date; echo MARK; grep -l FINAL_NCCL_DONE /tmp/final-nccl.log; grep -l FINISH_PPL_DONE /tmp/finish-ppl.log; grep -l Q8MMQ_AB_DONE /tmp/q8mmq-ab.log; echo TAIL; tail -3 /tmp/q8mmq-ab.log; tail -2 /tmp/finish-ppl.log" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
