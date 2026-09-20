@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\fa-raw-and-model.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "echo ===FARAWSHAPES===; grep -a -E 'FLASH_ATTN_EXT' /tmp/opbench-fa.txt | grep -a 'type_K=q8_0' | cut -c1-190; echo ===FARAWNUMS===; grep -a -B1 'us/run' /tmp/opbench-fa.txt | grep -a -E 'FLASH_ATTN_EXT|us/run' | grep -a -A1 'type_K=q8_0' | grep -a 'us/run' | cut -c1-120; echo ===MOSTLOGS===; ls -t /tmp/*server.log 2>/dev/null | head -3" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
