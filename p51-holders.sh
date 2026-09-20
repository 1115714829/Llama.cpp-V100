#!/bin/bash
# Who actually holds /dev/nvidia* ? (reveals orphaned/foreign holders that ps may not show)
set -uo pipefail
echo "=== fuser on nvidia device nodes ==="
for d in /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3 /dev/nvidia4 /dev/nvidia5 /dev/nvidiactl /dev/nvidia-uvm; do
  if [ -e "$d" ]; then
    printf '%-18s ' "$d"
    fuser "$d" 2>/dev/null || echo -n ""
    echo ""
  fi
done

echo ""
echo "=== PIDs with an nvidia fd open ==="
found=0
for p in /proc/[0-9]* ; do
  pid=${p#/proc/}
  if ls -l "$p/fd" 2>/dev/null | grep -q -e nvidia -e uvm ; then
    found=1
    echo "pid $pid : $(cat $p/comm 2>/dev/null) : $(ls -l $p/fd 2>/dev/null | grep -c -e nvidia -e uvm) nvidia fds"
  fi
done
[ "$found" = "0" ] && echo "(no process has an nvidia fd open)"

echo ""
echo "=== kernel modules / persistence ==="
lsmod 2>/dev/null | grep -e nvidia -e nvlink || echo "(lsmod unavailable)"
systemctl is-active nvidia-persistenced 2>/dev/null || true

echo ""
echo "=== current per-GPU memory ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo P51_DONE
