@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\nccl-pkg.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
set PKG=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime
echo === pkg tree === > "%OUT%"
%SSHH% "ls -lR %PKG%" >> "%OUT%" 2>&1
echo === nccl version macros === >> "%OUT%"
%SSHH% "grep -n 'NCCL_MAJOR\|NCCL_MINOR\|NCCL_PATCH\|NCCL_VERSION_CODE' %PKG%/include/nccl.h" >> "%OUT%" 2>&1
echo === exported nccl symbols === >> "%OUT%"
%SSHH% "nm -D --defined-only %PKG%/lib/libnccl.so.2" >> "%OUT%" 2>&1
echo === pkg metadata === >> "%OUT%"
%SSHH% "ls -l %PKG%.dist-info" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
