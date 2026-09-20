@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes mk-corpus.sh root@192.168.50.235:/root/mk-corpus.sh
%SSHH% "bash /root/mk-corpus.sh"
%SSHH% "tail -3 /tmp/nccl-ab-summary.log"
