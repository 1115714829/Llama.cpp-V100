#!/bin/bash
A=$(pgrep -f z1-arm | wc -l)
B=$(pgrep -f z1-harness | wc -l)
C=$(pgrep -x llama-server | wc -l)
D=$(pgrep -x llama-bench | wc -l)
if [ "$A" != "0" ] || [ "$B" != "0" ] || [ "$C" != "0" ] || [ "$D" != "0" ]; then
  echo "REFUSED_BUSY z1arm=$A z1harness=$B server=$C bench=$D"
  exit 3
fi
nohup bash /root/fak-build.sh > /root/fak-build-run.log 2>&1 < /dev/null &
echo "LAUNCHED pid=$! at $(date +%H:%M:%S)"
