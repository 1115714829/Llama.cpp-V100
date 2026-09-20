@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes nccl-knobs2.sh root@192.168.50.235:/root/nccl-knobs2.sh
%SSHH% "nohup bash /root/nccl-knobs2.sh > /tmp/nccl-knobs2.log 2>&1 < /dev/null & echo LAUNCHED"
%SSHH% "sleep 4; pgrep -af 'nccl-knobs2|nccl-env-sweep|final-nccl'"
%SSHH% "grep -aE 'MEDIAN_TG=' /tmp/p60-env-*.log 2>/dev/null"
