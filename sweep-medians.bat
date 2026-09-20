@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\sweep-medians.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === per-arm results (all arms) === > "%OUT%"
%SSHH% "for t in ab-bf-tp3 ab-nccl-tp3 ab-nccl-tp4 ab-nccl-tp6 ab-bf-layer ab-nccl-layer env-debug env-plain env-tp2 env-ring env-ll env-1ch env-ring1ch env-ringll; do printf '== %s : ' $t; grep -aE 'MEDIAN_TG=' /tmp/p60-$t.log 2>/dev/null | tr '\n' ' '; grep -aE '^prompt[123]: ' /tmp/p60-$t.log 2>/dev/null | tr '\n' ' '; echo; grep -a 'greedy:' /tmp/p60-$t.log 2>/dev/null | tail -1; done" >> "%OUT%" 2>&1
echo === sweep log tail === >> "%OUT%"
%SSHH% "tail -40 /tmp/nccl-env-sweep.log" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
