@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes q8mmq-ab.sh root@192.168.50.235:/root/q8mmq-ab.sh
%SSHH% "nohup bash /root/q8mmq-ab.sh > /tmp/q8mmq-ab.log 2>&1 < /dev/null & echo LAUNCHED"
%SSHH% "sleep 4; pgrep -af 'q8mmq-ab|collect2'"
