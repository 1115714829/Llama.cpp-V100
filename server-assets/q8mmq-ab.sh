#!/bin/bash
# q8mmq-ab.sh -- Volta Q8_0 at ne11=8: MMVQ (current) vs MMQ.
# Applies an exact-text patch to mmvq.{cuh,cu}, rebuilds build-nccl (same cmake config =>
# the only variable is the source diff), snapshots, then A/Bs at TP3 tensor.
set -u
SRC=/root/llm/test/v100-opt/llama.cpp
PKGLIB=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/lib
H=/root/p60-ab-harness2.sh
cd "$SRC" || exit 1

echo "=== waiting for the current queue (formal runs + ppl) ==="
for i in $(seq 1 200); do
  grep -q FINISH_PPL_DONE /tmp/finish-ppl.log 2>/dev/null && { echo "queue done"; break; }
  sleep 15
done

echo "=== patch ==="
python3 - <<'PY'
import sys
cuh = open('ggml/src/ggml-cuda/mmvq.cuh').read()
old_cuh = '#define MMVQ_VOLTA_MAX_BATCH_SIZE_K 4\n'
new_cuh = ('#define MMVQ_VOLTA_MAX_BATCH_SIZE_K 4\n\n'
           '// Same crossover for the dp4a legacy quants on Volta.  The default limit of 8 puts the\n'
           '// speculative verify batch (ne11 = 8) on MMVQ, which is the one shape this path serves.\n'
           '#define MMVQ_VOLTA_MAX_BATCH_SIZE_LEGACY 4\n')
assert cuh.count(old_cuh) == 1, 'cuh anchor not unique'
if 'MMVQ_VOLTA_MAX_BATCH_SIZE_LEGACY' not in cuh:
    open('ggml/src/ggml-cuda/mmvq.cuh','w').write(cuh.replace(old_cuh, new_cuh, 1))

cu = open('ggml/src/ggml-cuda/mmvq.cu').read()
old_cu = ('            case GGML_TYPE_Q4_K:\n'
          '                return ne11 <= MMVQ_VOLTA_MAX_BATCH_SIZE_K;\n'
          '            default:\n'
          '                return ne11 <= MMVQ_MAX_BATCH_SIZE;\n')
new_cu = ('            case GGML_TYPE_Q4_K:\n'
          '                return ne11 <= MMVQ_VOLTA_MAX_BATCH_SIZE_K;\n'
          '            case GGML_TYPE_Q8_0:\n'
          '                return ne11 <= MMVQ_VOLTA_MAX_BATCH_SIZE_LEGACY;\n'
          '            default:\n'
          '                return ne11 <= MMVQ_MAX_BATCH_SIZE;\n')
assert cu.count(old_cu) == 1, 'cu anchor not unique (count=%d)' % cu.count(old_cu)
if 'MMVQ_VOLTA_MAX_BATCH_SIZE_LEGACY' not in cu:
    open('ggml/src/ggml-cuda/mmvq.cu','w').write(cu.replace(old_cu, new_cu, 1))
print('patched ok')
PY
grep -n 'MMVQ_VOLTA_MAX_BATCH_SIZE_LEGACY' ggml/src/ggml-cuda/mmvq.cuh ggml/src/ggml-cuda/mmvq.cu

echo "=== rebuild build-nccl (same cmake config) ==="
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
cmake --build build-nccl --config Release -j82 2>&1 | tail -6
mkdir -p /root/libdir-nccl-q8mmq
cp -a build-nccl/bin/. /root/libdir-nccl-q8mmq/
ln -sf "$PKGLIB/libnccl.so.2" /root/libdir-nccl-q8mmq/libnccl.so.2
md5sum /root/libdir-nccl-q8mmq/libggml-cuda.so.0.24.0 /root/libdir-nccl/libggml-cuda.so.0.24.0

echo "=== A/B: MMVQ@8 (current) vs MMQ@8 (patched) ==="
sweep() { # <tag> <libdir>
  echo "##### Q8 ARM $1 libdir=$2"
  ( env NODROP=1 TAG="$1" L="$2" LDEXTRA="$PKGLIB" CARDS=0,1,2 SPLIT=tensor PORT=8170 P2P=1 \
      bash "$H" ) 2>&1 | tail -16
}
sweep q8-mmvq   /root/libdir-nccl
sweep q8-mmq    /root/libdir-nccl-q8mmq
sweep q8-mmvq2  /root/libdir-nccl
sweep q8-mmq2   /root/libdir-nccl-q8mmq
echo "=== medians ==="
for t in q8-mmvq q8-mmq q8-mmvq2 q8-mmq2; do
  printf '%-10s ' "$t"; grep -a -E "^prompt[123]: |^MEDIAN_TG=" /tmp/p60-$t.log 2>/dev/null | tr '\n' ' '; echo
done
echo Q8MMQ_AB_DONE
