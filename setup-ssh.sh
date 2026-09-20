#!/bin/bash
set -uo pipefail
SRC=/mnt/c/Users/a1115/.qwen/skills/ssh-server/id_rsa_pem
AC=192.168.50.235
echo "=== source key ==="
ls -la "$SRC" 2>&1
mkdir -p ~/.ssh && chmod 700 ~/.ssh
cp "$SRC" ~/.ssh/id_rsa
chmod 600 ~/.ssh/id_rsa
echo "=== install host key ==="
ssh-keyscan -t ed25519,rsa "$AC" >> ~/.ssh/known_hosts 2>/dev/null
chmod 644 ~/.ssh/known_hosts 2>/dev/null
ls -la ~/.ssh/id_rsa
echo "=== test key auth + GPU2/5 state ==="
ssh -o BatchMode=yes -o ConnectTimeout=10 root@$AC 'echo SKILL_KEY_OK; date; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader' 2>&1 | head -14
echo DONE
