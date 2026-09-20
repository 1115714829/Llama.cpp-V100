@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\cards2.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do if grep -q CARDS2_DONE /tmp/cards2.log 2>/dev/null; then break; fi; sleep 30; done; grep -aE '^##### CARDS ARM|^TAG=|^prompt[123]: |^MEDIAN_TG=|via |isAllDirectP2p|connected all rings|Connected all rings|NVLS|Using network|^  GPU|NV[0-9]|SYS|target decode\+sync|CARDS2_DONE' /tmp/cards2.log" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
