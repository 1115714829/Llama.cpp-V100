#!/bin/bash
# Pair the per-op timing lines (perf output) with the op names from the export file, by index.
set -u
for UB in 2048 512; do
    OPS=/tmp/ops-ub$UB.txt
    PERF=/tmp/perf-ub$UB.txt
    [ -f "$OPS" ] || continue
    [ -f "$PERF" ] || continue
    echo "===== -ub $UB : $(wc -l < "$OPS") ops exported, $(wc -l < "$PERF") timed ====="
    paste -d'@' <(awk '{print $NF}' "$OPS") "$PERF" \
        | awk -F'@' '{split($2,a," "); v=a[4]+0; u=a[5]; if (u=="ms/run") v=v*1000; printf "%12.1f us  %s\n", v, $1}' \
        | sort -k1 -gr | head -18
    echo ""
done
echo "===== full list, -ub 2048 ====="
paste -d'@' <(awk '{print $NF}' /tmp/ops-ub2048.txt) /tmp/perf-ub2048.txt \
    | awk -F'@' '{split($2,a," "); v=a[4]+0; u=a[5]; if (u=="ms/run") v=v*1000; printf "%12.1f us  %s\n", v, $1}' \
    | sort -k1 -gr | head -45
