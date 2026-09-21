#!/bin/bash
# z1-launch.sh -- preflight (machine quiet, port free) then nohup one Z1 arm.
set -uo pipefail
S=$(pgrep -f 'llama-serve[r] --model' | wc -l)
SX=$(pgrep -x llama-server | wc -l)
B=$(pgrep -f 'llama-benc[h] -m' | wc -l)
BX=$(pgrep -x llama-bench | wc -l)
F=$(pgrep -f fa-correctnes[s] | wc -l)
echo PREFLIGHT time=$(date +%H:%M:%S) srv_f=$S srv_x=$SX bench_f=$B bench_x=$BX fa=$F
if [ $S -ne 0 ] || [ $SX -ne 0 ] || [ $B -ne 0 ] || [ $BX -ne 0 ] || [ $F -ne 0 ]; then
  echo PREFLIGHT_BUSY
  exit 1
fi
if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
  echo PORT_BUSY port=$PORT
  exit 1
fi
echo PORT_FREE port=$PORT
export CTX TAG PORT USESPEC
nohup bash /root/z1-arm.sh > /tmp/z1-$TAG-driver.log 2>&1 < /dev/null &
echo LAUNCHED pid=$!
