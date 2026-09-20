#!/bin/bash
echo "HOME=$HOME"
echo "=== tools ==="
for t in sshpass expect ssh-keygen ssh scp; do
  if command -v "$t" >/dev/null; then echo "$t: $(command -v $t)"; else echo "$t: MISSING"; fi
done
echo "=== ssh dir ==="
ls -la ~/.ssh/ 2>&1
echo "=== try ssh key auth ==="
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@192.168.50.235 'echo KEY_OK' 2>&1 | head -2
echo CHECK_DONE
