#!/bin/bash
# nccl-env-sweep.sh -- after the first NCCL A/B finishes, sweep NCCL transport/protocol knobs
# (the round does ~128 allreduces, so small-message latency dominates) plus TP2.
set -u
PKGLIB=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/lib
H=/root/p60-ab-harness2.sh
echo "=== waiting for first NCCL A/B to finish ==="
for i in $(seq 1 120); do
  grep -q NCCL_AB_DONE /tmp/nccl-ab.log 2>/dev/null && { echo "previous sweep done"; break; }
  sleep 15
done
grep -q NCCL_AB_DONE /tmp/nccl-ab.log || echo "WARN: previous sweep still not done; continuing anyway"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

sweep() { # <tag> <cards> [ENV=VAL ...]
  tag=$1; cards=$2; shift 2
  echo "##### ENV ARM $tag cards=$cards env='$*'"
  ( export "$@"; env NODROP=1 TAG="$tag" L=/root/libdir-nccl LDEXTRA="$PKGLIB" \
      CARDS="$cards" SPLIT=tensor PORT=8150 P2P=1 bash "$H" ) 2>&1 | tail -20
}

sweep env-debug    0,1,2 NCCL_DEBUG=INFO
sweep env-plain    0,1,2
sweep env-tp2      0,1   NCCL_DEBUG=WARN
sweep env-ring     0,1,2 NCCL_ALGO=Ring
sweep env-ll       0,1,2 NCCL_PROTO=LL,Simple
sweep env-1ch      0,1,2 NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 NCCL_NTHREADS=128
sweep env-ring1ch  0,1,2 NCCL_ALGO=Ring NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1
sweep env-ringll   0,1,2 NCCL_ALGO=Ring NCCL_PROTO=LL,Simple NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1

echo "=== NCCL init banner (proof it is really NCCL, and which transport) ==="
grep -a "NCCL INFO" /tmp/p60-env-debug-server.log | head -25
echo "=== per-arm medians ==="
for t in env-debug env-plain env-tp2 env-ring env-ll env-1ch env-ring1ch env-ringll; do
  printf '%-12s ' "$t"
  grep -a -E "^prompt[123]: |^MEDIAN_TG=" /tmp/p60-$t.log 2>/dev/null | tr '\n' ' '
  echo
done
echo NCCL_ENV_SWEEP_DONE
