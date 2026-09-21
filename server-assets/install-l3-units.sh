#!/bin/bash
# Install L3 acceptance units for the production model at 256K, one per variant.
# 3 cards inside NUMA node 0 (0,1,2): 17.23 GiB weights + ~14 GiB KV/scratch at 256k
# => ~10.4 GiB/card, and no cross-NUMA all-reduce.
# Sampling params = HuggingFace generation_config.json of the TurboFCFusion repo
#   (temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05) per user direction.
# MTP is OFF (--spec-type none): we know the draft sampler falls back to CPU under
# --split-mode tensor (see mtp-sampler-cpu.md), so a clean baseline excludes it.
set -uo pipefail

MODEL=/root/llm/models/Qwen3.8-27B-TurboFCFusion-gguf/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf

write_unit() {
  local name="$1" libdir="$2" port="$3" alias="$4" sysid="$5"
  cat > "/etc/systemd/system/${name}.service" <<EOF
[Unit]
Description=llama.cpp TurboFCFusion Q4_K_M 256K L3 (${alias})
Wants=network-online.target
After=network-online.target nvidia-persistenced.service

[Service]
Type=simple
User=root
WorkingDirectory=/root/llm/llama.cpp
Environment="CUDA_VISIBLE_DEVICES=0,1,2"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64:${libdir}"
ExecStart=${libdir}/llama-server \\
  --model ${MODEL} \\
  --alias ${alias} \\
  --ctx-size 262144 \\
  --n-gpu-layers 999 \\
  --split-mode tensor --tensor-split 1,1,1 \\
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \\
  --parallel 1 \\
  --spec-type none \\
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 \\
  --presence-penalty 0.0 --repeat-penalty 1.05 \\
  --reasoning on --reasoning-preserve \\
  --slot-save-path /home/llama-slot-cache \\
  --host 0.0.0.0 --port ${port} --metrics \\
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
SyslogIdentifier=${sysid}

[Install]
WantedBy=multi-user.target
EOF
  echo "wrote /etc/systemd/system/${name}.service (libdir=${libdir} port=${port})"
}

write_unit "llama-server-l3-pristine" "/root/libdir-pristine" 8083 "qwen38-tfcf-q4km-l3-pristine" "llama-l3-pristine"
write_unit "llama-server-l3-c4"       "/root/libdir-c4"       8084 "qwen38-tfcf-q4km-l3-c4"       "llama-l3-c4"

systemctl daemon-reload
echo ""
echo "=== installed (NOT started) ==="
systemctl list-unit-files 2>/dev/null | grep -E "llama-server-l3" || echo "(not found)"
echo ""
echo "=== verify the two libdirs are the distinct variants ==="
md5sum /root/libdir-pristine/libggml-cuda.so.0.24.0 /root/libdir-c4/libggml-cuda.so.0.24.0
echo ""
echo "=== model present? ==="
ls -la "$MODEL"
echo ""
echo "=== current GPU state (must be free before starting) ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo L3_UNITS_DONE
