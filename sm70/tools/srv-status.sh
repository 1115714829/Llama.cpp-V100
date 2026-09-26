#!/bin/bash
# srv-status.sh - read-only snapshot of the AC922 test server. Changes nothing.
# Usage (from the workstation): scp this file to /root/llm/test/, then
#   ssh -o BatchMode=yes root@192.168.50.235 'bash /root/llm/test/srv-status.sh'
redact() { sed -E 's/(--api-key[= ]+)[^ ]+/\1<REDACTED>/g; s/(([A-Z_]*(KEY|TOKEN|SECRET|PASSWORD)[A-Z_]*)=)[^ "]+/\1<REDACTED>/g'; }

echo "== host";     cat /proc/sys/kernel/hostname; uptime
echo "== gpu";      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
echo "== gpu_apps"; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
echo "== services"
for s in vllm-1cat llmscope new-api llama-server; do printf '%s=' "$s"; systemctl is-active "$s"; done
echo "== tmux";     tmux ls 2>&1
echo "== procs";    pgrep -af 'llama-serve[r]|vll[m]|llama-benc[h]|test-backend-op[s]|cmake --buil[d]|nvc[c] ' | cut -c1-220 | redact
echo "== locks";    ls -l /tmp/LLAMA_BUILD_LOCK /root/llm/test/AGENT_LOCK 2>&1
echo "== disk";     df -h / /mnt/3.84t 2>&1
echo "== mem";      free -g
echo "== models"
ls -la /mnt/3.84t/Qwen3.8-27B-GGUF/ 2>&1
ls /mnt/3.84t/llm-models/ 2>&1
echo "== tree"
ls /root/llm/test/v100-opt/llama.cpp 2>&1 | head -40
ls -la --time-style=+%F_%T /root/llm/test/v100-opt/llama.cpp/build-instr/bin 2>&1 | head -40
echo "== libdir_gb"
ls -la --time-style=+%F_%T /root/libdir-gb 2>&1 | head -40
cat /root/libdir-gb/BUILD_MANIFEST.txt 2>&1
echo "== test_dir"
ls -la --time-style=+%F_%T /root/llm/test/*.sh /root/llm/test/*.py /root/llm/test/*.txt 2>&1 | head -80
echo "== vllm_unit"
systemctl cat vllm-1cat 2>&1 | redact | head -80
echo "== cuda";     /usr/local/cuda/bin/nvcc --version 2>&1 | tail -2; ls -d /usr/local/cuda* 2>&1
echo "== topo";     nvidia-smi topo -m 2>&1
echo STATUS_DONE
