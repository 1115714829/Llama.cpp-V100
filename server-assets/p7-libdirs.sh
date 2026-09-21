#!/bin/bash
# P7: save the PRISTINE library dir, then rebuild C4 in place (so we can snapshot C4 libs too).
# Lesson: llama-bench/llama-server are launcher stubs (209944 B); the code is in libggml-cuda.so (~125 MB).
# DT_RUNPATH points at the build dir absolutely, and LD_LIBRARY_PATH takes precedence over DT_RUNPATH,
# so per-variant lib dirs + LD_LIBRARY_PATH give a real A/B.
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp

echo "=== [1] save PRISTINE libdir (stub + all shared libs) ==="
rm -rf /root/libdir-pristine
mkdir -p /root/libdir-pristine
cp -a "$SRC/build/bin/." /root/libdir-pristine/
echo "files: $(ls /root/libdir-pristine | wc -l)"
md5sum /root/libdir-pristine/libggml-cuda.so.0.24.0 /root/libdir-pristine/libllama.so.0.4.1
echo ""

echo "=== [2] remove the useless stub-only copies from before ==="
rm -f /root/bin-b11053-c4-server /root/bin-b11053-c4-bench /root/bin-b11053-pristine-server /root/bin-b11053-pristine-bench
echo "removed (they were byte-identical stubs)"
echo ""

echo "=== [3] swap in C4 mmvq.cu + rebuild in place ==="
cp /root/mmvq-c4-backup.cu "$SRC/ggml/src/ggml-cuda/mmvq.cu"
echo "C4 refs in source: $(grep -c MMVQ_PARAMETERS_VOLTA "$SRC/ggml/src/ggml-cuda/mmvq.cu")"
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j8 --target llama-server llama-bench" > /tmp/build-c4b.log 2>&1 &
echo "launched PID=$! log=/tmp/build-c4b.log"
sleep 10
tail -3 /tmp/build-c4b.log
echo P7_DONE
