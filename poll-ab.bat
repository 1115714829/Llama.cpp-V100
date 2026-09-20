@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\nccl-ab-progress.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === build verdict (tail) === > "%OUT%"
%SSHH% "tail -22 /tmp/build-nccl.log" >> "%OUT%" 2>&1
echo === ab running? === >> "%OUT%"
%SSHH% "pgrep -af 'nccl-ab|llama-server'" >> "%OUT%" 2>&1
echo === libdir-nccl contents === >> "%OUT%"
%SSHH% "ls -l /root/libdir-nccl/" >> "%OUT%" 2>&1
echo === ab log so far === >> "%OUT%"
%SSHH% "cat /tmp/nccl-ab.log" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
