#!/bin/bash
# T1-C runner. Launches the SELF-BUILT launcher + libs (LIBS, default
# /root/libdir-gb) so no production binary or production lib dir is involved.
# Service shape matches the BL2/BL3 baselines: 4 cards, tensor split, 256K ctx,
# ub 2048, q8_0 KV, spec off, alias Qwen3.8-27B-Q8_0-BL2.
#   TAG=dq1 T1C_DUMP=1 BLOCKING=1 UB=2048 bash t1c-run.sh
set -u
TAG=${TAG:-dq1}
LOG=/root/llm/test/$TAG.log
: > "$LOG"
if pgrep -f 'llama-serve[r] --model' >/dev/null; then echo "SERVER_ALREADY_RUNNING"; exit 2; fi
LIBS=${LIBS:-/root/libdir-gb}
export LD_LIBRARY_PATH=$LIBS:/usr/local/cuda/lib64
export CUDA_VISIBLE_DEVICES=${CARDS:-0,1,2,3}
export GGML_GALLOCR_SLOTS=${SLOTS:-1}
if [ "${NO79T:-}" != "1" ]; then export LLAMA_SM70_79T=1; fi
SPEC_ARGS=""
if [ "${SPEC:-}" = "1" ]; then
  SPEC_ARGS="--model-draft /mnt/3.84t/llm-models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max ${NMAX:-7}"
else
  export NO_SPEC=1
fi
if [ -n "${ROUND:-}" ]; then export LLAMA_ROUND_TIMING=1; fi
if [ -n "${T1C_DUMP:-}" ]; then export T1C_DUMP=1; fi
if [ -n "${T1C_REF:-}" ]; then export T1C_REF=1; fi
if [ -n "${T1C_REF_PREFIX_ONLY:-}" ]; then export T1C_REF_PREFIX_ONLY=1; fi
if [ -n "${T1C_SKIP_TAIL:-}" ]; then export T1C_SKIP_TAIL=1; fi
if [ -n "${T1C_OWSUM:-}" ]; then export T1C_OWSUM=1; fi
if [ -n "${T1C_REF_MAX:-}" ]; then export T1C_REF_MAX=$T1C_REF_MAX; fi
if [ -n "${T1C_STAGE_STOP:-}" ]; then export T1C_STAGE_STOP=$T1C_STAGE_STOP; fi
if [ -n "${BLOCKING:-}" ]; then export CUDA_LAUNCH_BLOCKING=1; fi
if [ -n "${DECOMP:-}" ]; then export LLAMA_SM70_FA_DECOMP=$DECOMP; fi
echo "RUN tag=$TAG bin=$LIBS/llama-server cards=$CUDA_VISIBLE_DEVICES ub=${UB:-2048} owsum=${T1C_OWSUM:-0} $(date)" | tee -a "$LOG"
exec "$LIBS/llama-server" \
  --model /mnt/3.84t/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf \
  --alias Qwen3.8-27B-Q8_0-BL2 \
  --ctx-size 262144 \
  --n-gpu-layers 999 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 --ubatch-size "${UB:-2048}" \
  $SPEC_ARGS \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --reasoning on --reasoning-preserve \
  --slot-save-path /home/llama-slot-cache \
  --host 0.0.0.0 --port 8082 --metrics >> "$LOG" 2>&1
