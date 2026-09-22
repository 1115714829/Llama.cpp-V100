#!/bin/bash
# Mechanism test for the long-context prefill wall: with quantized KV, the TILE/MMA_F16 flash
# attention path converts the WHOLE K and V to f16 on every ubatch (fattn-common.cuh:1026-1088),
# so the cost is O(n_kv) per ubatch = O(N^2/ub) overall. With f16 KV there is no conversion at all.
#
# Same build, same cards, same ub: only --cache-type-k/v differ. Depth series in ONE load per arm.
set -u

SRC=/root/llm/test/v100-opt/llama.cpp
LAB=/mnt/3.84t/v100-opt/kernel-lab
BIN=$SRC/build-instr/bin
M=/mnt/3.84t/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
LOG=$LAB/cachetype-experiment.log
MARK=$LAB/cachetype-experiment.done

rm -f "$MARK"
exec >"$LOG" 2>&1
echo "=== START $(date -Is)"
export CUDA_VISIBLE_DEVICES=0,1,2
export GGML_CUDA_P2P=1

fail() { echo "ABORT_$1"; rm -f /tmp/LLAMA_BUILD_LOCK; touch "$MARK"; exit 1; }
pgrep -f 'llama-serve[r] --model' >/dev/null && fail BUSY_SERVER
pgrep -f 'cmake --buil[d]'       >/dev/null && fail BUSY_CMAKE
[ -e /tmp/LLAMA_BUILD_LOCK ] && fail BUILD_LOCK
touch /tmp/LLAMA_BUILD_LOCK

echo "lib: $(md5sum $BIN/libggml-cuda.so.0.24.0)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

run() {   # $1 = tag, $2 = ctk/ctv type, $3 = depths
    echo ""
    echo "=== ARM $1  cache-type=$2  depths=$3  $(date -Is)"
    # hot page cache keeps the per-arm cost to the measurement itself (diagnostic calibre: NODROP)
    "$BIN/llama-bench" -m "$M" -p "$3" -n 0 -r 3 -ub 2048 -ts 1/1/1 \
        --flash-attn 1 -ctk "$2" -ctv "$2" 2>&1 | grep -v -e '^ggml_cuda' -e '^load_' -e '^main:'
    echo "--- ARM $1 done $(date -Is)"
}

run q8_0 "q8_0" "8192,32768,131072"
run f16  "f16"  "8192,32768,131072"

echo ""
echo "=== DONE $(date -Is)"
rm -f /tmp/LLAMA_BUILD_LOCK
touch "$MARK"
