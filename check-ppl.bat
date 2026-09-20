@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\ppl-tooling.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === perplexity binaries in libdirs === > "%OUT%"
%SSHH% "ls -l /root/libdir-rt/llama-perplexity /root/libdir-rt/llama-bench /root/libdir-goal-base/llama-perplexity" >> "%OUT%" 2>&1
echo === candidate corpus files === >> "%OUT%"
%SSHH% "ls -l /root/llm/test/v100-opt/llama.cpp/tests /root/llm/models" >> "%OUT%" 2>&1
echo === build still running? === >> "%OUT%"
%SSHH% "pgrep -af build-nccl-inner; tail -3 /tmp/build-nccl.log" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
