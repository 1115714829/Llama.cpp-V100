#!/bin/bash
# Gate: wait for the Q8_0 sha256, and only launch the baseline if it matches exactly.
set -uo pipefail
EXP=a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348

echo "=== waiting for sha256 to finish (29 GB read over NFS) ==="
n=0
while pgrep -f sha256sum >/dev/null 2>&1; do
  sleep 20; n=$((n+20))
  if [ $((n % 120)) -eq 0 ]; then echo "  ...still hashing ${n}s"; fi
  if [ "$n" -gt 2100 ]; then echo "sha timeout - giving up"; break; fi
done

echo ""
echo "=== sha256 result ==="
tail -4 /tmp/sha-q8.log

echo ""
if grep -q "$EXP" /tmp/sha-q8.log; then
  echo "SHA256 MATCH -> launching Phase 0.3 baseline"
  sed -i 's/\r$//' /root/p35-base-q8.sh /root/wait-base-q8.sh
  setsid nohup bash /root/p35-base-q8.sh < /dev/null > /tmp/p35.out 2>&1 &
  echo "baseline launched"
else
  echo "SHA256 MISMATCH or absent -> NOT launching the baseline"
fi
echo P36_DONE
