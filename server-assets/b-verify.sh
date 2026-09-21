#!/bin/bash
# Post-build real-machine verification battery
set -u
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
L=/root/libdir-inst
export GGML_CUDA_P2P=1
echo === 1 regression: harness, instrumented lib, env OFF ===; date
CARDS=0,1,2 SPLIT=tensor TAG=instr-reg L=$L P2P=1 NODROP=1 PORT=8140 bash /root/p60-ab-harness.sh
echo === 2 instrumented round timing: sync + splits ===; date
LLAMA_ROUND_TIMING_SYNC=1 CARDS=0,1,2 SPLIT=tensor TAG=instr-sync L=$L P2P=1 NODROP=1 PORT=8141 bash /root/p60-ab-harness.sh
echo === 3 per-op + allreduce counters, graphs OFF ===; date
GGML_CUDA_OP_TIMING=1 NOGRAPH=1 CARDS=0,1,2 SPLIT=tensor TAG=instr-op L=$L P2P=1 NODROP=1 PORT=8142 bash /root/p60-ab-harness.sh
echo === 4 M1 prefill: TP3 vs TP4 (f16 KV, zero code) ===; date
for TS in 1/1/1 1/1/1/1; do
  case $TS in 1/1/1) CD=0,1,2;; *) CD=0,1,2,3;; esac
  echo --- cards=$CD ts=$TS ---
  LD_LIBRARY_PATH=$L CUDA_VISIBLE_DEVICES=$CD $L/llama-bench -m $M -ngl 999 -fa on -sm tensor -ts $TS -r 2 -o md -p 32768 -n 8 -ctk f16 -ctv f16 2>&1 | grep -a -e pp32768 -e tg8 | tail -2
done
echo === 5 M2 prefill: ub 512 vs 2048 (TP4) ===; date
for UB in 512 2048; do echo --- ub=$UB ---; LD_LIBRARY_PATH=$L CUDA_VISIBLE_DEVICES=0,1,2,3 $L/llama-bench -m $M -ngl 999 -fa on -sm tensor -ts 1/1/1/1 -r 2 -o md -p 32768 -n 8 -ub $UB -ctk f16 -ctv f16 2>&1 | grep -a -e pp32768 -e tg8 | tail -2; done
echo BVERIFY_ALL_DONE; date