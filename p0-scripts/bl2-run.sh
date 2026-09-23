#!/bin/bash
# BL2/BL3 runner: the llama-server-bl2.service ExecStart with a selectable lib set.
#   LIBS=""            -> official binary's own libs (BL2a, literal baseline)
#   LIBS=/root/libdir-pristine -> b11053 stock libs (BL2b, same-source fairness)
#   LIBS=/root/libdir-t8b      -> B5 adopted chain (BL3)
# SLOTS=3 is set for the B5 lib set (R282: 8-slot default OOMs at >=32K shapes).
set -u
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=${LIBS:+$LIBS:}/usr/local/cuda/lib64
if [ -n "${SLOTS:-}" ]; then export GGML_GALLOCR_SLOTS=$SLOTS; fi
SPEC_ARGS=""
if [ "${NO_SPEC:-}" != "1" ]; then
  SPEC_ARGS="--model-draft /mnt/3.84t/llm-models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7"
fi
exec /root/llm/llama.cpp/bin/llama-server \
  --model /mnt/3.84t/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf \
  --alias Qwen3.8-27B-Q8_0-BL2 \
  --ctx-size 262144 \
  --n-gpu-layers 999 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  $SPEC_ARGS \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --reasoning on --reasoning-preserve \
  --slot-save-path /home/llama-slot-cache \
  --host 0.0.0.0 --port 8082 --metrics
