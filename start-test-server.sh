#!/bin/bash
set -uo pipefail
# Stop the stuck original server (free GPU + port). Do NOT modify the original unit file.
echo "=== stop stuck llama-server (original, TurboFCFusion) ==="
systemctl stop llama-server 2>/dev/null
sleep 5
systemctl is-active llama-server
echo "=== write NEW unit: llama-server-test.service (our test model, 256k, port 8081) ==="
cat > /etc/systemd/system/llama-server-test.service <<'UNIT'
[Unit]
Description=llama.cpp Qwen3.8-27B Q2_K_XL Test Model Server (256k)
Wants=network-online.target
After=network-online.target nvidia-persistenced.service

[Service]
Type=simple
User=root
WorkingDirectory=/root/llm/llama.cpp
Environment="CUDA_VISIBLE_DEVICES=2,5"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64"
ExecStart=/root/llm/llama.cpp/bin/llama-server \
  --model /root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf \
  --alias Qwen3.8-27B-Q2_K_XL \
  --ctx-size 262144 \
  --n-gpu-layers 999 \
  --split-mode tensor --tensor-split 1,1 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --reasoning on --reasoning-preserve \
  --host 0.0.0.0 --port 8081 --metrics \
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
SyslogIdentifier=llama-server-test

[Install]
WantedBy=multi-user.target
UNIT
echo "unit written"
systemctl daemon-reload
echo "=== start llama-server-test ==="
systemctl start llama-server-test
sleep 6
systemctl is-active llama-server-test
echo "=== GPU 2/5 ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 5
echo "=== journalctl command (dynamic) ==="
echo "journalctl -u llama-server-test -f"
echo TEST_START_DONE
