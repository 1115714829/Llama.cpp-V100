@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes microbench.sh root@192.168.50.235:/root/microbench.sh
%SSHH% "nohup bash /root/microbench.sh > /tmp/microbench.log 2>&1 < /dev/null & echo LAUNCHED"
%SSHH% "sleep 5; pgrep -af 'test-backend-ops|microbench'"
