#!/bin/bash
# build.sh - fail-closed build (main agent only). The tree must equal the uploaded workstation manifest,
# the output goes to a NEW lib dir (a lib dir in use is never overwritten), and BUILD_MANIFEST.txt records provenance.
# env: COMMIT (required: git short hash the manifest was made from), TREE, OUT_ROOT
set -uo pipefail
COMMIT=${COMMIT:?set COMMIT=<git short hash of the workstation tree>}
TREE=${TREE:-/root/llm/test/v100-opt/llama.cpp}
OUT_ROOT=${OUT_ROOT:-/mnt/3.84t/sm70/libs}
W=/root/llm/test/sm70
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=$OUT_ROOT/$COMMIT-$STAMP
BLOG=$W/logs/build-$STAMP.log

[ -e /tmp/LLAMA_BUILD_LOCK ] && { echo "BUILD_REFUSED lock /tmp/LLAMA_BUILD_LOCK"; exit 2; }
if pgrep -f 'llama-serve[r] --model|test-backend-op[s] |cmake --buil[d]' > /dev/null; then echo "BUILD_REFUSED busy"; exit 2; fi
touch /tmp/LLAMA_BUILD_LOCK
trap 'rm -f /tmp/LLAMA_BUILD_LOCK' EXIT

bash $W/tools/tree-check.sh || { echo "BUILD_REFUSED tree_drift"; exit 3; }
cd "$TREE" || exit 2
cmake --build build-instr -j128 --target llama-server llama-bench llama-perplexity test-backend-ops test-export-graph-ops > "$BLOG" 2>&1
RC=$?
echo "BUILD_RC=$RC log=$BLOG"
if [ "$RC" != "0" ]; then grep -a -m 30 -e 'error' "$BLOG"; exit 1; fi

mkdir -p "$OUT"
cp -a build-instr/bin/. "$OUT/"
{
  echo "COMMIT=$COMMIT"
  echo "BUILD_TIME=$(date -Iseconds)"
  echo "TREE=$TREE"
  echo "MANIFEST_SHA256=$(sha256sum $W/manifest-server.cmp | cut -c1-64)"
  echo "MANIFEST_FILES=$(wc -l < $W/manifest-server.cmp)"
  for f in libggml-cuda.so.0.24.0 libggml-base.so.0.24.0 libllama.so.0.4.1 libllama-common.so.0.4.1 llama-server test-backend-ops; do
    echo "MD5_$f=$(md5sum "$OUT/$f" 2>/dev/null | cut -c1-32)"
  done
  grep -E '^(CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|GGML_CUDA[A-Z_]*|GGML_NCCL[A-Z_]*):' build-instr/CMakeCache.txt
} > "$OUT/BUILD_MANIFEST.txt"
cat "$OUT/BUILD_MANIFEST.txt"
echo "BUILD_OUT=$OUT"
