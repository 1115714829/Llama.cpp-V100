@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\model-params.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "echo ===LOGS===; ls -t /tmp/p60*server.log | head -5; echo ===PARAMS===; for f in /tmp/p60-ab-nccl-tp3-server.log /tmp/p60-ab-nccl-layer-server.log /tmp/p60-ab-bf-tp3-server.log /tmp/p60-c4-debug-server.log; do if [ -f $f ]; then echo FILE=$f; grep -a -E 'block_count|head_count|embedding_length|feed_forward_length|key_length|value_length|ssm\.|shortconv|n_expert|expert_count|context_length|rope' $f | head -35; break; fi; done" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
