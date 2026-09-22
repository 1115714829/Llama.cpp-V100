#!/bin/bash
# Profile the mma prototype: is the scattered (lane = row) weight load the limiter?
set -u
NCU=/usr/local/cuda-12.4/bin/ncu
LAB=/mnt/3.84t/v100-opt/kernel-lab
export CUDA_VISIBLE_DEVICES=0
METRICS=dram__bytes.sum,lts__t_sectors.sum,l1tex__t_sectors.sum,smsp__inst_executed.sum,gpu__time_duration.sum

for NS in 1 8; do
    echo "===== prototype nsplit=$NS ====="
    $NCU --metrics "$METRICS" --kernel-name regex:w8volta_split \
        --launch-skip 5 --launch-count 1 \
        "$LAB/w8volta-bench" 4096 14336 $NS 20 2>&1 | grep -e "Metric Value" -e dram__bytes -e gpu__time -e l1tex__ -e lts__ -e smsp__inst
done

echo "===== prototype nsplit=8 stalls ====="
$NCU --section WarpStateStats --section SchedulerStats --kernel-name regex:w8volta_split \
    --launch-skip 5 --launch-count 1 \
    "$LAB/w8volta-bench" 4096 14336 8 20 2>&1 | grep -e "Warp Cycles" -e "Active Warps" -e "Eligible" -e "Issued Warp" -e "Est. Speedup" -e "stalled waiting" -e "L1TEX" -e "L1 instruction"
echo "=== DONE ==="
