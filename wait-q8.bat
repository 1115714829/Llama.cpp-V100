@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\q8.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60; do if grep -q Q8MMQ_AB_DONE /tmp/q8mmq-ab.log 2>/dev/null; then break; fi; sleep 30; done; echo ===Q8LOG===; grep -aE 'patch|patched|MMVQ_VOLTA_MAX_BATCH_SIZE_LEGACY|Built target|error|medians|^q8-|ARM|MEDIAN_TG=|Q8MMQ_AB_DONE|libggml-cuda.so' /tmp/q8mmq-ab.log; echo ===FORMAL2===; grep -aE 'MEDIAN_TG=|^prompt[123]:|^greedy:|FINAL_NCCL_DONE' /tmp/final-nccl.log; echo ===PPL===; grep -aE 'corpus:|PPL ARM|estimate|FINISH_PPL_DONE' /tmp/finish-ppl.log" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
