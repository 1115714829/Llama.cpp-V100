#!/bin/bash
set -uo pipefail
echo "=== stop current 256k curl + test server ==="
pkill -f "curl.*8081" 2>/dev/null
systemctl stop llama-server-test 2>/dev/null
sleep 6
systemctl is-active llama-server-test
echo "=== add MTP 4 to test unit (spec-type draft-mtp, spec-draft-n-max 4) ==="
# Insert the MTP args right after the --parallel line in the unit file.
sed -i 's|  --parallel 1 \\|  --parallel 1 \\\n  --spec-type draft-mtp --spec-draft-n-max 4 \\|' /etc/systemd/system/llama-server-test.service
echo "--- unit MTP line ---"
grep -n "spec-type\|parallel" /etc/systemd/system/llama-server-test.service
systemctl daemon-reload
echo "=== restart test server (with MTP 4) ==="
systemctl start llama-server-test
sleep 6
systemctl is-active llama-server-test
echo "=== journalctl command ==="
echo "journalctl -u llama-server-test -f"
echo MTP_RESTART_DONE
