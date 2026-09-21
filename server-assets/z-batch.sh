#!/bin/bash
# z-batch.sh -- the remaining zero-code experiments in one command.
#
# Default is DIAG mode (NODROP=1): each arm loads in about 15 s instead of 255 s
# (see AGENTS.md item 34). DIAG numbers are for DIRECTION ONLY and must not be quoted.
#   NODROP=0 bash /root/z-batch.sh     -> official protocol, 255 s per arm
#
# Arms:
#   Z3  n_max sweep at ctx 8192 (3 / 5 / 7): is our n_max=7 still right at short ctx?
#   Z7  model-size scaling, no speculation: Q8_0 (27 GiB) vs Q2_K_XL (9.14 GiB).
#       Linear scaling in weight bytes => purely bandwidth bound (BW1 is the lever);
#       super-linear => there is a weight-independent fixed cost (N6/P-B first).
set -uo pipefail

D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
BIG=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
SMALL=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q2_K_XL.gguf
L=${L:-/root/libdir-instr}
PORT=${PORT:-8291}
NODROP=${NODROP:-1}
NPRED=${NPRED:-128}
OUT=${OUT:-/tmp/z-batch.txt}
: > "$OUT"

if [ "$NODROP" = "1" ]; then
  echo "MODE: DIAG (NODROP=1) - direction only, NOT for quoted numbers" | tee -a "$OUT"
else
  echo "MODE: OFFICIAL (drop_caches each arm, ~255 s per arm)" | tee -a "$OUT"
fi

run() {
  local tag=$1; shift
  local p=$PORT; PORT=$((PORT+1))
  echo "" | tee -a "$OUT"
  echo "=== $tag ===" | tee -a "$OUT"
  env CARDS=0,1,2 SPLIT=tensor L="$L" P2P=1 NPRED="$NPRED" TAG="$tag" PORT="$p" NODROP="$NODROP" "$@" \
      bash /root/p60-ab-harness.sh 2>&1 | grep -a -e tg= -e MEDIAN_TG -e STATUS -e health | tee -a "$OUT"
  echo "ARM_RC=$?" | tee -a "$OUT"
}

run z3-n3 SPEC="--model-draft $D --spec-type draft-dflash --spec-draft-n-max 3"
run z3-n5 SPEC="--model-draft $D --spec-type draft-dflash --spec-draft-n-max 5"
run z3-n7 SPEC="--model-draft $D --spec-type draft-dflash --spec-draft-n-max 7"
run z7-big   M="$BIG"   SPEC="--spec-type none"
run z7-small M="$SMALL" SPEC="--spec-type none"

echo "" | tee -a "$OUT"
echo "DONE_Z_BATCH" | tee -a "$OUT"
