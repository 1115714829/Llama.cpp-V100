@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\eol-check.txt
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes md5-eol-check.sh root@192.168.50.235:/root/md5-eol-check.sh
echo === server: lf vs crlf md5 === > "%OUT%"
%SSHH% "bash /root/md5-eol-check.sh" >> "%OUT%" 2>&1
echo === local: git ls-files --eol === >> "%OUT%"
cd /d F:\vllm+llama.cpp\llama.cpp
git ls-files --eol common/speculative.cpp src/llama-context.cpp ggml/src/ggml-cuda/mmvq.cu src/llama.cpp common/common.cpp >> "%OUT%" 2>&1
echo === local: core.autocrlf === >> "%OUT%"
git config --get core.autocrlf >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
