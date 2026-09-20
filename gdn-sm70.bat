@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\gdn-sm70.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
cd /d F:\vllm+llama.cpp\1cat-vllm
echo === files mentioning FUSED_GDN / gdn verify === > "%OUT%"
%SSHH% "grep -rIl 'FUSED_GDN\|fused_gdn' /root/llm/../../ 2>/dev/null | head -5" >> "%OUT%" 2>&1
echo === sm70 gdn/attention dirs in 1cat === >> "%OUT%"
dir /s /b csrc\flash_qla 2>nul >> "%OUT%"
echo === FUSED_GDN_VERIFY refs === >> "%OUT%"
findstr /s /i /m /c:"FUSED_GDN_VERIFY" *.* >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
