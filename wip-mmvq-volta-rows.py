#!/usr/bin/env python3
# R244: Volta MMVQ rows_per_block for ncols_dst 5..8. usage: patch-rows.py <ggml-cuda dir> <rows>
# Bounded, uniqueness-enforced, bytes-only, CRLF aware. Prints LINES_ADDED for diff verification.
import hashlib, sys, pathlib
ROOT = pathlib.Path(sys.argv[1]); RW = int(sys.argv[2])
F = ROOT / 'mmvq.cu'
NL = b'\r\n'
def md5(): return hashlib.md5(F.read_bytes()).hexdigest()
b = F.read_bytes()
print('MD5_BEFORE=' + md5())
SIG = b'static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {'
ANCHOR = NL.join([SIG, b'    if (table_id == MMVQ_PARAMETERS_GENERIC'])
n = b.count(ANCHOR)
print('ANCHOR_HITS=' + str(n))
if n != 1:
    print('ANCHOR_NOT_UNIQUE'); sys.exit(2)
if b'MMVQ_PARAMETERS_VOLTA && ncols_dst >= 5' in b:
    print('ALREADY_APPLIED'); print('MD5_AFTER=' + md5()); sys.exit(0)
NEW = NL.join([SIG,
               b'    if (table_id == MMVQ_PARAMETERS_VOLTA && ncols_dst >= 5 && ncols_dst <= 8) {',
               b'        return ' + str(RW).encode() + b';',
               b'    }'])
b2 = b.replace(ANCHOR, NEW + NL + b'    if (table_id == MMVQ_PARAMETERS_GENERIC', 1)
F.write_bytes(b2)
print('LINES_ADDED=' + str(b2.count(NL) - b.count(NL)))
print('MD5_AFTER=' + md5())
print('PATCHED_ROWS=' + str(RW))
