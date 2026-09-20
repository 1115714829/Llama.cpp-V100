@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === before ===
%SSHH% "pgrep -af 'kno[b]s2|finis[h]-ppl|pp[l]-after-final|fina[l]-nccl|nc[c]l-env-sweep'"
echo === kill old ppl waiter ===
%SSHH% "kill 1171713 2>&1; sleep 1; pgrep -af 'finis[h]-ppl'"
echo === launch new ppl waiter ===
%SSHH% "nohup bash /root/ppl-after-final.sh >> /tmp/finish-ppl.log 2>&1 < /dev/null & echo LAUNCHED"
%SSHH% "sleep 3; pgrep -af 'pp[l]-after-final'"
