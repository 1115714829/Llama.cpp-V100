@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\hmbench.txt
scp -o BatchMode=yes hm_q8_bench.cu run-hm-bench.sh root@192.168.50.235:/root/
%SSHH% "bash /root/run-hm-bench.sh" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
