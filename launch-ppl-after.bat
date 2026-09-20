@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes ppl-after-final.sh root@192.168.50.235:/root/ppl-after-final.sh
%SSHH% "pkill -f nccl-knobs2.sh; pkill -f finish-ppl.sh; sleep 2; nohup bash /root/ppl-after-final.sh >> /tmp/finish-ppl.log 2>&1 < /dev/null & echo LAUNCHED"
%SSHH% "sleep 4; pgrep -af 'ppl-after-final|nccl-knobs2|finish-ppl|final-nccl|nccl-env-sweep' | head -8"
