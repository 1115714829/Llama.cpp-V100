#!/bin/bash
# z1-depth.sh DEPTH [REPS] [TAG] -- one llama-bench real-depth arm with full hygiene.
set -uo pipefail
DEPTH=$1
REPS=${2:-1}
TAG=${3:-$DEPTH}
OUT=/tmp/z1-depth-$TAG.log
export LD_LIBRARY_PATH=/root/libdir-instr
S=$(pgrep -f 'llama-serve[r] --model' | wc -l)
B=$(pgrep -f 'llama-benc[h]' | wc -l)
F=$(pgrep -f fa-correctnes[s] | wc -l)
echo "PREFLIGHT depth=$DEPTH reps=$REPS tag=$TAG time=$(date +%H:%M:%S) srv=$S bench=$B fa=$F" | tee "$OUT"
if [ $S -ne 0 ] || [ $B -ne 0 ] || [ $F -ne 0 ]; then echo PREFLIGHT_BUSY | tee -a "$OUT"; exit 1; fi
sync; echo 3 > /proc/sys/vm/drop_caches; sleep 3
echo "gpu mem before:" | tee -a "$OUT"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$OUT"
echo "BENCH_CMD: /root/libdir-instr/llama-bench -m Qwen3.8-27B-Q8_0.gguf -ngl 999 -sm tensor -ts 1/1/1 -p 0 -n 32 -d $DEPTH -ctk q8_0 -ctv q8_0 -fa 1 -ub 2048 -r $REPS -o md" | tee -a "$OUT"
/root/libdir-instr/llama-bench -m /root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf -ngl 999 -sm tensor -ts 1/1/1 -p 0 -n 32 -d "$DEPTH" -ctk q8_0 -ctv q8_0 -fa 1 -ub 2048 -r "$REPS" -o md 2>&1 | tee -a "$OUT"
RC=${PIPESTATUS[0]}
echo "BENCH_RC=$RC" | tee -a "$OUT"
echo "gpu mem after:" | tee -a "$OUT"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$OUT"
echo "DEPTH_DONE depth=$DEPTH reps=$REPS tag=$TAG time=$(date +%H:%M:%S)" | tee -a "$OUT"
