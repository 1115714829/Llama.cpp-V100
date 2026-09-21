#!/bin/bash
# devd-retest.sh -- is --spec-draft-device still crashing?  If it works, the draft stops paying
# the tensor-split allreduce (14.4 ms -> ~4.4 ms per round, ~9 ms of the ~59 ms round).
set -u
PKGLIB=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/lib
H=/root/p60-ab-harness2.sh
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

sweep() { # <tag> <devd> [extra env...]
  tag=$1; devd=$2; shift 2
  echo "##### DEVD ARM $tag devd='$devd' env='$*'"
  ( export "$@"; env NODROP=1 TAG="$tag" L=/root/libdir-nccl LDEXTRA="$PKGLIB" \
      CARDS=0,1,2 SPLIT=tensor PORT=8195 P2P=1 DEVD="$devd" bash "$H" ) 2>&1 | tail -20
}

sweep devd-none   ""
sweep devd-cuda0  "CUDA0"
sweep devd-cuda1  "CUDA1"
sweep devd-none2  ""

echo "=== medians + draft phase ==="
for t in devd-none devd-cuda0 devd-cuda1 devd-none2; do
  printf '%-11s ' "$t"
  grep -a -E "^prompt[123]: |^MEDIAN_TG=" /tmp/p60-$t.log 2>/dev/null | tr '\n' ' '
  echo
  grep -a "spec timing" /tmp/p60-$t-server.log 2>/dev/null | tail -1
  grep -aE "GGML_ASSERT|Aborted|ggml_abort|not supported" /tmp/p60-$t-server.log 2>/dev/null | tail -2
done
echo DEVD_RETEST_DONE
