#!/bin/bash
# T1-C runner: thin env wrapper over the adopted-chain BL2 launcher (bl2-run.sh),
# so the T1-C A/B sits on exactly the same service shape as the ledger baseline.
#   TAG=dq4 T1C_DUMP=1 BLOCKING=1 UB=2048 bash t1c-run.sh
set -u
TAG=${TAG:-dq4}
LOG=/root/llm/test/$TAG.log
: > "$LOG"
if pgrep -f 'llama-serve[r] --model' >/dev/null; then echo "SERVER_ALREADY_RUNNING"; exit 2; fi
if [ "${NO79T:-}" != "1" ]; then export LLAMA_SM70_79T=1; fi
export NO_SPEC=1
export UB=${UB:-2048}
export LIBS=${LIBS:-/root/libdir-gb}
export SLOTS=${SLOTS:-1}
if [ -n "${T1C_DUMP:-}" ]; then export T1C_DUMP=1; fi
if [ -n "${BLOCKING:-}" ]; then export CUDA_LAUNCH_BLOCKING=1; fi
if [ -n "${T1C_STAGE_STOP:-}" ]; then export T1C_STAGE_STOP=$T1C_STAGE_STOP; fi
if [ -n "${DECOMP:-}" ]; then export LLAMA_SM70_FA_DECOMP=$DECOMP; fi
echo "RUN tag=$TAG ub=$UB slots=$SLOTS dump=${T1C_DUMP:-0} blocking=${BLOCKING:-0} stop=${T1C_STAGE_STOP:-0} $(date)" | tee -a "$LOG"
exec bash /root/llm/test/bl2-run.sh >> "$LOG" 2>&1
