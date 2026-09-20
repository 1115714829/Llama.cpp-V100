@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes devd-retest.sh root@192.168.50.235:/root/devd-retest.sh
%SSHH% "nohup bash /root/devd-retest.sh > /tmp/devd-retest.log 2>&1 < /dev/null & echo LAUNCHED"
%SSHH% "sleep 5; pgrep -af 'devd-retest'"
