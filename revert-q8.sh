#!/bin/bash
# revert-q8.sh -- undo the falsified Q8_0->MMQ change on the server tree (measured -31%).
set -u
cd /root/llm/test/v100-opt/llama.cpp || exit 1
python3 - <<'PY'
cuh = open('ggml/src/ggml-cuda/mmvq.cuh').read()
blk = ('\n// Same crossover for the dp4a legacy quants on Volta.  The default limit of 8 puts the\n'
       '// speculative verify batch (ne11 = 8) on MMVQ, which is the one shape this path serves.\n'
       '#define MMVQ_VOLTA_MAX_BATCH_SIZE_LEGACY 4\n')
if blk in cuh:
    open('ggml/src/ggml-cuda/mmvq.cuh','w').write(cuh.replace(blk, '', 1))
    print('cuh reverted')
else:
    print('cuh already clean')

cu = open('ggml/src/ggml-cuda/mmvq.cu').read()
add = ('            case GGML_TYPE_Q8_0:\n'
       '                return ne11 <= MMVQ_VOLTA_MAX_BATCH_SIZE_LEGACY;\n')
if add in cu:
    open('ggml/src/ggml-cuda/mmvq.cu','w').write(cu.replace(add, '', 1))
    print('cu reverted')
else:
    print('cu already clean')
PY
echo "=== residual references (expect none) ==="
grep -c 'MMVQ_VOLTA_MAX_BATCH_SIZE_LEGACY' ggml/src/ggml-cuda/mmvq.cuh ggml/src/ggml-cuda/mmvq.cu
echo "=== server md5 (LF) ==="
md5sum ggml/src/ggml-cuda/mmvq.cuh ggml/src/ggml-cuda/mmvq.cu
echo "=== as CRLF (to compare with the Windows tree) ==="
sed 's/$/\r/' ggml/src/ggml-cuda/mmvq.cuh | md5sum
sed 's/$/\r/' ggml/src/ggml-cuda/mmvq.cu | md5sum
echo "=== ppl both arms ==="
grep -aE 'Final estimate|error|failed' /tmp/ppl-bf.log | tail -3
grep -aE 'Final estimate|error|failed' /tmp/ppl-nccl.log | tail -3
echo REVERT_DONE
