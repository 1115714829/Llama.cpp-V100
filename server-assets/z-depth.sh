#!/bin/bash
# z-depth.sh -- context-depth scan for decode, with an explicit time/quality tradeoff.
#
# Why: --ctx-size only ALLOCATES the KV; it does not FILL it (AGENTS.md item 33).
#      llama-bench -d really fills it.
#      And an arm's wall clock is 85-95 percent model loading, so diagnostic runs
#      must use the fast path (AGENTS.md item 34).
#
# DIAG=1 (default): no drop_caches -> load is page-cache warm (~15 s). DIAGNOSTIC ONLY.
# DIAG=0          : drop_caches each arm -> the official protocol. Costs ~255 s per arm.
#
# Usage:  DIAG=1 bash /root/z-depth.sh
set -uo pipefail

L=${L:-/root/libdir-instr}
M=${M:-/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf}
N=${N:-32}
R=${R:-2}
DIAG=${DIAG:-1}
OUT=${OUT:-/tmp/z-depth.txt}
DEPTHS=${DEPTHS:-8192 32768 131072}
: > "$OUT"

if [ "$DIAG" = "1" ]; then
  MODE="DIAG (no drop_caches, NOT valid for quoted numbers)"
else
  MODE="OFFICIAL (drop_caches per arm)"
fi
echo "z-depth: mode=$MODE lib=$L reps=$R depths=$DEPTHS" | tee -a "$OUT"

for DEPTH in $DEPTHS; do
  if pgrep -f 'llama-serve[r] --model' >/dev/null; then
    echo "BUSY: llama-server running - aborting (measurement must own the machine)" | tee -a "$OUT"; exit 1
  fi
  if pgrep -f 'llama-benc[h] -m' >/dev/null; then
    echo "BUSY: llama-bench running - aborting" | tee -a "$OUT"; exit 1
  fi
  echo "" | tee -a "$OUT"
  echo "=== depth=$DEPTH ($MODE) ===" | tee -a "$OUT"
  if [ "$DIAG" != "1" ]; then
    sync
    echo 3 > /proc/sys/vm/drop_caches
  fi
  LD_LIBRARY_PATH="$L" "$L/llama-bench" -m "$M" -ngl 999 -sm tensor -ts 1/1/1 \
      -p 0 -n "$N" -d "$DEPTH" -ctk q8_0 -ctv q8_0 -fa 1 -ub 2048 -r "$R" -o md \
      >> "$OUT" 2>&1
  echo "ARM_RC=$?" | tee -a "$OUT"
done

echo "" | tee -a "$OUT"
echo "=== result rows ===" | tee -a "$OUT"
grep -a -e "|" "$OUT" | grep -a -e "t/s" | tee -a "$OUT"
echo "DONE_Z_DEPTH" | tee -a "$OUT"
