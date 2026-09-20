#!/bin/bash
set -uo pipefail
AC=192.168.50.235
PW='fei15614132590'

echo "=== ensure pip ==="
if ! python3 -m pip --version >/dev/null 2>&1; then
  echo "pip missing; trying ensurepip --user"
  python3 -m ensurepip --user 2>&1 | tail -2
fi
python3 -m pip --version 2>&1 | head -1

echo "=== ensure paramiko (user install) ==="
if ! python3 -c "import paramiko" 2>/dev/null; then
  echo "paramiko missing; installing"
  python3 -m pip install --user --upgrade paramiko 2>&1 | tail -4
fi
python3 -c "import paramiko; print('paramiko', paramiko.__version__)" 2>&1 | head -1

echo "=== generate ssh key ==="
mkdir -p ~/.ssh && chmod 700 ~/.ssh
if [ ! -f ~/.ssh/id_ed25519 ]; then
  ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N '' -q
fi
PUB=$(cat ~/.ssh/id_ed25519.pub)
echo "PUB=$PUB"

echo "=== authorize key on AC922 via paramiko ==="
python3 - "$AC" "$PW" "$PUB" <<'PYEOF'
import sys, paramiko
ac, pw, pub = sys.argv[1], sys.argv[2], sys.argv[3]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(ac, username='root', password=pw, timeout=25)
cmds = [
    "mkdir -p ~/.ssh && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys",
    "grep -qF \"%s\" ~/.ssh/authorized_keys 2>/dev/null || echo \"%s\" >> ~/.ssh/authorized_keys" % (pub, pub),
    "wc -l ~/.ssh/authorized_keys",
]
for cmd in cmds:
    stdin, stdout, stderr = c.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print("OUT:", out)
    if err:
        print("ERR:", err)
c.close()
print("PARAMIKO_DONE")
PYEOF

echo "=== test key auth ==="
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@$AC 'echo KEY_AUTH_OK; date' 2>&1 | head -3
echo DONE
