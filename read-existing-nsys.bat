@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes read-existing-nsys.sh root@192.168.50.235:/root/read-existing-nsys.sh
%SSHH% "bash /root/read-existing-nsys.sh" > F:\vllm+llama.cpp\1cat-vllm-v100-study\existing-kern.txt 2>&1
echo DONE >> F:\vllm+llama.cpp\1cat-vllm-v100-study\existing-kern.txt
