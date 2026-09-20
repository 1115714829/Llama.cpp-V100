@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\arm1.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "grep -aE '^TAG=|^config:|MEDIAN_TG=|^prompt[123]:|^greedy:|perf:|spec timing|target decode\+sync|health ok|allreduce init' /tmp/p60-final-bf-tp3.log" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
