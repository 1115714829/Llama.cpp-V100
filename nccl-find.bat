@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\nccl-find.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === find in /root === > "%OUT%"
%SSHH% "find /root -name 'libnccl.so*'" >> "%OUT%" 2>&1
echo === find nccl.h === >> "%OUT%"
%SSHH% "find /root /usr/local /opt -name 'nccl.h'" >> "%OUT%" 2>&1
echo === find in /usr === >> "%OUT%"
%SSHH% "find /usr -maxdepth 7 -name 'libnccl.so*'" >> "%OUT%" 2>&1
echo === pip nccl === >> "%OUT%"
%SSHH% "pip3 list" >> "%OUT%" 2>&1
echo === pip3 list done === >> "%OUT%"
%SSHH% "ls -d /root/*venv* /root/*env* /opt/*venv*" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
