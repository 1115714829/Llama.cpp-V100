#!/bin/bash
# shape-filter.sh <in> <out> <op_id> - keep the test-file lines of one ggml op (first column = op id).
IN=${1:?usage: shape-filter.sh <in> <out> <op_id>}
OUT=${2:?usage: shape-filter.sh <in> <out> <op_id>}
OP=${3:?usage: shape-filter.sh <in> <out> <op_id>}
awk -v id="$OP" '$1 == id' "$IN" > "$OUT"
echo "SHAPES op=$OP lines=$(wc -l < "$OUT") out=$OUT"
