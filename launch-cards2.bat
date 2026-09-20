@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes cards2.sh root@192.168.50.235:/root/
%SSHH% "nohup bash /root/cards2.sh > /tmp/cards2.log 2>&1 < /dev/null & echo LAUNCHED"
%SSHH% "sleep 6; pgrep -af 'cards2|llama-server' | head -4"
