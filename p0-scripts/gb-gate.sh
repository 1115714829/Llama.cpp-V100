#!/bin/bash
# Gate arms: greedy sha256 must reproduce f3edac19... with bucket OFF and ON
# (the bit-equal claim of graph-shape-bucket.md [S2]). P60 fixed-yardstick harness.
set -u
run() {
  tag=$1
  bucket=$2
  spec=${3:-}
  export GGML_KV_BUCKET_RATIO=$bucket
  if [ -n "$spec" ]; then export SPEC="$spec"; else unset SPEC; fi
  echo "=== ARM $tag bucket='$bucket' spec='${spec:-default}' $(date) ==="
  CARDS=0,1,2 SPLIT=tensor L=/root/libdir-gb P2P=1 TAG=$tag NPRED=512 NODROP=1 PORT=8331 \
    bash /root/p60-ab-harness.sh > /tmp/gb-gate-$tag.log 2>&1
  echo "rc=$?"
  grep -iE 'sha|gate|draft_n|draft_acc' /tmp/gb-gate-$tag.log | tail -5
}
run gb-off ''
run gb-off-nospec '' "--spec-type none"
run gb-on 1.25
run gb-on-nospec 1.25 "--spec-type none"
echo GATE_DONE
