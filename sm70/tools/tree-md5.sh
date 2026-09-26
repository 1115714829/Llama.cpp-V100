#!/bin/bash
# tree-md5.sh <tree> <out> - "<md5>  <relpath>" for every source/build-config file, CR stripped, byte-order sorted.
# Must produce byte-identical output to tree-md5.ps1 on the workstation. Top-level build*/ and .git are skipped.
set -euo pipefail
TREE=${1:?usage: tree-md5.sh <tree> <out>}
OUT=${2:?usage: tree-md5.sh <tree> <out>}
cd "$TREE"
find . \( -path './build*' -o -path './.git' \) -prune -o -type f \
  \( -name '*.c' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.hpp' \
     -o -name '*.cmake' -o -name 'CMakeLists.txt' \) -print \
  | sed 's|^\./||' | LC_ALL=C sort | while IFS= read -r f; do
    printf '%s  %s\n' "$(tr -d '\r' < "$f" | md5sum | cut -c1-32)" "$f"
  done > "$OUT"
echo "TREE_MD5 files=$(wc -l < "$OUT") out=$OUT"
