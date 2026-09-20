#!/bin/bash
set -uo pipefail
C4BIN=/root/llm/test/v100-opt/llama.cpp/build/bin/llama-server
echo "=== stop original test server (port 8081) ==="
pkill -f "curl.*8081" 2>/dev/null
systemctl stop llama-server-test 2>/dev/null
sleep 6
systemctl is-active llama-server-test
echo "=== write C4 unit: llama-server-c4.service (C4 binary, same config, port 8082) ==="
cat > /etc/systemd/system/llama-server-c4.service <<UNIT
[Unit]
Description=llama.cpp C4 Qwen3.8-27B Q2_K_XL Test Model Server (256k, MTP4)
Wants=network-online.target
After=network-online.target nvidia-persistenced.service

[Service]
Type=simple
User=root
WorkingDirectory=/root/llm/llama.cpp
Environment="CUDA_VISIBLE_DEVICES=2,5"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64"
ExecStart=$C4BIN \\
  --model /root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf \\
  --alias Qwen3.8-27B-Q2_K_XL-C4 \\
  --ctx-size 262144 \\
  --n-gpu-layers 999 \\
  --split-mode tensor --tensor-split 1,1 \\
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \\
  --parallel 1 \\
  --spec-type draft-mtp --spec-draft-n-max 4 \\
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \\
  --presence-penalty 0.0 --repeat-penalty 1.0 \\
  --reasoning on --reasoning-preserve \\
  --slot-save-path /home/llama-slot-cache \\
  --host 0.0.0.0 --port 8082 --metrics \\
  --api-key-file /etc/llama-server/api-keys
Restart=on-failure
RestartSec=10s
TimeoutStartSec=300s
TimeoutStopSec=60s
KillSignal=SIGTERM
LimitNOFILE=1048576
TasksMax=infinity
StandardOutput=journal
StandardError=journal
SyslogIdentifier=llama-server-c4

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
echo "=== start C4 server ==="
systemctl start llama-server-c4
for i in $(seq 1 15); do
  sleep 6
  P=$(ss -ltn 2>/dev/null | grep -c 8082)
  if [ "$P" -gt 0 ]; then echo "PORT_8082_UP after $((i*6))s"; break; fi
done
systemctl is-active llama-server-c4
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 5
journalctl -u llama-server-c4 -n 40 --no-pager 2>/dev/null | grep -iE "model loaded|listening|backend offload|spec" | tail -5
echo "=== journalctl command (C4) ==="
echo "journalctl -u llama-server-c4 -f"
echo "=== IMMEDIATELY launch 256k TTFT test (C4, port 8082) ==="
sed 's/8081/8082/g; s/stress-orig/stress-c4/g' /root/send-256k.sh > /root/send-256k-c4.sh
bash /root/send-256k-c4.sh
echo C4_START_AND_BENCH_DONE
