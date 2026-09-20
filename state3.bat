@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\state3.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "pgrep -af 'fina[l]-nccl|pp[l]-after-final|llama-perplex[ity]|llama-serve[r]' | head -5" > "%OUT%" 2>&1
echo === FINAL === >> "%OUT%"
%SSHH% "grep -aE 'ARM|^TAG=|^config:|^libs:|^prompt[123]:|^MEDIAN_TG=|^greedy:|perf:|spec timing|target decode\+sync|FINAL_NCCL_DONE' /tmp/final-nccl.log" >> "%OUT%" 2>&1
echo === PPL === >> "%OUT%"
%SSHH% "grep -aE 'corpus:|PPL ARM|Final estimate|exit=|libggml-cuda|FINISH_PPL_DONE' /tmp/finish-ppl.log" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
