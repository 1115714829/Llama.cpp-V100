#!/bin/bash
# cards2.sh -- re-test the card-count question with NCCL, including 1cat's own card set, and
# find out what transport NCCL actually picks across the two NUMA groups.
set -u
PKGLIB=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/lib
H=/root/p60-ab-harness2.sh
echo "=== topology ==="
nvidia-smi topo -m 2>&1 | head -12 | tail -8
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

run() { # <tag> <cards> <port> [env...]
  tag=$1; cards=$2; port=$3; shift 3
  echo "##### CARDS ARM $tag cards=$cards"
  ( export "$@"; env NODROP=1 TAG="$tag" L=/root/libdir-nccl LDEXTRA="$PKGLIB" \
      CARDS="$cards" SPLIT=tensor PORT="$port" P2P=1 bash "$H" ) 2>&1 | tail -14
}

run c3-012     0,1,2     8200
run c4-0123    0,1,2,3   8201
run c4-0134    0,1,3,4   8202
run c5-01234   0,1,2,3,4 8203
run c6-012345  0,1,2,3,4,5 8204
# NCCL transport decision at 4 ranks
run c4-debug   0,1,2,3   8205 NCCL_DEBUG=INFO

echo "=== NCCL transport lines at TP4 (NVLink vs SYS/host) ==="
grep -aiE "via (P2P|SYS|NVLS)|P2P/direct|Connected all rings|isAllDirectP2p|Setting affinity|Channel .*: 0|Using network" \
  /tmp/p60-c4-debug-server.log 2>/dev/null | head -30
echo "=== medians + per-round ms ==="
for t in c3-012 c4-0123 c4-0134 c5-01234 c6-012345; do
  printf '%-11s ' "$t"
  grep -a -E "^prompt[123]: |^MEDIAN_TG=" /tmp/p60-$t.log 2>/dev/null | tr '\n' ' '
  echo
  grep -a "target decode+sync" /tmp/p60-$t-server.log 2>/dev/null | tail -2
done
echo CARDS2_DONE
