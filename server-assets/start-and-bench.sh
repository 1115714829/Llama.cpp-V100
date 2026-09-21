#!/bin/bash
set -uo pipefail
# Add --slot-save-path to the unit (right after the --reasoning line), if not present.
if ! grep -q "slot-save-path" /etc/systemd/system/llama-server-test.service; then
  sed -i 's|  --reasoning on --reasoning-preserve \\|  --reasoning on --reasoning-preserve \\\n  --slot-save-path /home/llama-slot-cache \\|' /etc/systemd/system/llama-server-test.service
fi
echo "=== current ExecStart args ==="
grep -E "model |ctx-size|n-gpu-layers|split-mode|tensor-split|flash-attn|cache-type|parallel|spec-type|reasoning|slot-save-path|port " /etc/systemd/system/llama-server-test.service
systemctl daemon-reload
echo "=== restart test server ==="
systemctl restart llama-server-test
# Wait for the model to load + port 8081 up (poll up to 90s).
for i in $(seq 1 15); do
  sleep 6
  P=$(ss -ltn 2>/dev/null | grep -c 8081)
  if [ "$P" -gt 0 ]; then echo "PORT_8081_UP after $((i*6))s"; break; fi
done
echo "=== service + GPU ==="
systemctl is-active llama-server-test
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 5
echo "=== model loaded? (journal) ==="
journalctl -u llama-server-test -n 40 --no-pager 2>/dev/null | grep -iE "model loaded|listening|backend offload|spec" | tail -6
echo "=== IMMEDIATELY launch 256k TTFT test ==="
bash /root/send-256k.sh
echo START_AND_BENCH_DONE
