#!/bin/bash
# z1-ladder.sh TAG DEPTHS REPS KV -- dense depth ladder, one model load, full hygiene.
set -uo pipefail
TAG=$1
DEPTHS=$2
REPS=${3:-8}
KV=${4:-q8_0}
OUT=/tmp/z1-ladder-$TAG.log
export LD_LIBRARY_PATH=/root/libdir-instr
export GGML_CUDA_FA_KERNEL_DEBUG=1
export CUDA_VISIBLE_DEVICES=0,1,2
S=$(pgrep -f 'llama-serve[r] --model' | wc -l)
B=$(pgrep -f 'llama-benc[h]' | wc -l)
F=$(pgrep -f fa-correctnes[s] | wc -l)
M=$(pgrep -f 'cmake --buil[d]' | wc -l)
L=0; [ -e /tmp/LLAMA_BUILD_LOCK ] && L=1
echo "PREFLIGHT tag=$TAG depths=$DEPTHS reps=$REPS kv=$KV cvd=0,1,2 time=$(date +%H:%M:%S) srv=$S bench=$B fa=$F cmake=$M lock=$L" | tee "$OUT"
if [ $S -ne 0 ] || [ $B -ne 0 ] || [ $F -ne 0 ] || [ $M -ne 0 ] || [ $L -ne 0 ]; then echo PREFLIGHT_BUSY | tee -a "$OUT"; exit 1; fi
sync; echo 3 > /proc/sys/vm/drop_caches; sleep 3
echo "gpu mem before:" | tee -a "$OUT"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$OUT"
/root/libdir-instr/llama-bench -m /root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf -ngl 999 -sm tensor -ts 1/1/1 -p 0 -n 32 -d "$DEPTHS" -ctk "$KV" -ctv "$KV" -fa 1 -ub 2048 -r "$REPS" -o md 2>&1 | tee -a "$OUT"
RC=${PIPESTATUS[0]}
echo "BENCH_RC=$RC" | tee -a "$OUT"
echo "gpu mem after:" | tee -a "$OUT"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$OUT"
echo "LADDER_DONE tag=$TAG time=$(date +%H:%M:%S)" | tee -a "$OUT"