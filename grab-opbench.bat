@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes root@192.168.50.235:/tmp/opbench-gdn.txt ./opbench-gdn-raw.txt
scp -o BatchMode=yes root@192.168.50.235:/tmp/opbench-fa.txt ./opbench-fa-raw.txt
scp -o BatchMode=yes root@192.168.50.235:/tmp/opbench-mm.txt ./opbench-mm-raw.txt
scp -o BatchMode=yes root@192.168.50.235:/tmp/p60-ab-nccl-tp3-server.log ./p60-ab-nccl-tp3-server.log
%SSHH% "echo ===STATUS===; cat /tmp/opbench.log; echo ===MMLINES===; wc -l < /tmp/opbench-mm.txt" > opbench-final-status.txt 2>&1
echo DONE >> opbench-final-status.txt
