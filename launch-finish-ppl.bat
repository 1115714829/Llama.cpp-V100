@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes finish-ppl.sh root@192.168.50.235:/root/finish-ppl.sh
%SSHH% "nohup bash /root/finish-ppl.sh > /tmp/finish-ppl.log 2>&1 < /dev/null & echo LAUNCHED"
%SSHH% "sleep 4; pgrep -af 'finish-ppl|nccl-knobs2|final-nccl|nccl-env-sweep'"
