@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\prefill.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
set G=grep -aE "prompt eval time|eval time|tokens per second|n_prompt|n_ctx"
echo === final NCCL arm (drop_caches) === > "%OUT%"
%SSHH% "grep -aE 'prompt eval time|^ *eval time|slot print_timing.*n_ctx|n_prompt' /tmp/p60-final-nccl-tp3-server.log | head -24" >> "%OUT%" 2>&1
echo === final butterfly arm === >> "%OUT%"
%SSHH% "grep -aE 'prompt eval time' /tmp/p60-final-bf-tp3-server.log | head -12" >> "%OUT%" 2>&1
echo === devd-none (NODROP, latest) === >> "%OUT%"
%SSHH% "grep -aE 'prompt eval time' /tmp/p60-devd-none-server.log | head -12" >> "%OUT%" 2>&1
echo === ctx settings actually used === >> "%OUT%"
%SSHH% "grep -aE 'n_ctx|n_batch|n_ubatch' /tmp/p60-final-nccl-tp3-server.log | head -8" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
