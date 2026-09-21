#!/bin/sh
echo "=== whoami / host ==="
id
uname -a
cat /etc/redhat-release 2>/dev/null
echo "PATH=$PATH"
echo "=== uptime / load ==="
uptime
echo "=== cpu / mem (free -g) ==="
nproc
free -g
echo "=== disk ==="
df -h / /tmp 2>/dev/null
echo "=== gpu ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv
echo "=== services ==="
for s in vllm-1cat llmscope new-api llama-server llama-server-c4 llama-server-test; do
  printf "%-20s %s\n" "$s" "$(systemctl is-active $s 2>/dev/null)"
done
echo "=== project paths ==="
ls -d /root/llm/test/v100-opt/llama.cpp /root/llm/models /root/libdir-nccl 2>/dev/null
echo "=== toolchain ==="
/usr/local/cuda-12.4/bin/nvcc --version | tail -2
/opt/rh/gcc-toolset-12/root/usr/bin/gcc --version | head -1
echo "=== done ==="
