@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\block.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "sleep 560; date; echo MARK; grep -c FINAL_NCCL_DONE /tmp/final-nccl.log; grep -c FINISH_PPL_DONE /tmp/finish-ppl.log; grep -c Q8MMQ_AB_DONE /tmp/q8mmq-ab.log; echo FINAL2; grep -aE '^ARM|^prompt[123]:|^MEDIAN_TG=|^greedy:' /tmp/final-nccl.log; echo PPL; grep -aE 'corpus:|PPL ARM|estimate' /tmp/finish-ppl.log; echo Q8; tail -4 /tmp/q8mmq-ab.log" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
