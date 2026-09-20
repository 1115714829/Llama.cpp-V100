@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\final2.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === A) card sweep medians + per-round target forward === > "%OUT%"
%SSHH% "grep -aE '^##### CARDS ARM|^MEDIAN_TG=|prompt[123]: tg' /tmp/cards2.log" >> "%OUT%" 2>&1
echo === B) NCCL P2P verdict at 3 vs 4 cards === >> "%OUT%"
%SSHH% "grep -a 'isAllDirectP2p' /tmp/cards2.log | head -4; grep -a 'isAllDirectP2p' /tmp/p60-c3-012-server.log 2>/dev/null | head -2" >> "%OUT%" 2>&1
echo === C) our quant type q8_0: n=1 vs n=8 === >> "%OUT%"
%SSHH% "grep -a -A1 'type_a=q8_0,type_b=f32,m=4096,n=1,k=14336\|type_a=q8_0,type_b=f32,m=4096,n=8,k=14336' /tmp/mb-mulmat.log | grep -a 'us/run'" >> "%OUT%" 2>&1
echo === D) my best HMMA prototype === >> "%OUT%"
%SSHH% "CUDA_VISIBLE_DEVICES=0 /root/wmma_perf3 5120 17408 10 | grep -E 'us/run|GB/s'" >> "%OUT%" 2>&1
echo === E) 1cat unit flags (read-only) === >> "%OUT%"
%SSHH% "systemctl cat vllm-1cat | grep -aE 'CUDA_VISIBLE_DEVICES|tensor-parallel-size|gpu-memory-utilization|kv-cache-dtype|max-model-len'" >> "%OUT%" 2>&1
%SSHH% "grep -acE 'NCCL_' /root/llm/systemd/1cat-runtime.env" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
cd /d F:\vllm+llama.cpp\llama.cpp
echo === F) git status --porcelain === >> "%OUT%"
git status --porcelain >> "%OUT%" 2>&1
echo DONE2 >> "%OUT%"
