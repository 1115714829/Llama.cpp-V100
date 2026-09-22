#!/usr/bin/env python3
# R237: Volta MMVQ nwarps for ncols_dst 5..8. usage: patch-nwarps.py <ggml-cuda dir> <nwarps>
import hashlib, sys, pathlib
ROOT = pathlib.Path(sys.argv[1]); NW = int(sys.argv[2])
F = ROOT / 'mmvq.cu'
NL = b'\r\n'
def md5(): return hashlib.md5(F.read_bytes()).hexdigest()
b = F.read_bytes()
print('MD5_BEFORE=' + md5())
i = b.find(b'MMVQ_PARAMETERS_VOLTA) {')
if i < 0:
    print('NO_VOLTA_BRANCH'); sys.exit(2)
tpl = NL.join([b'            case 5:', b'            case 6:', b'            case 7:', b'            case 8:', b'                return X;'])
found = None
for v in (1, 2, 4, 8):
    cand = tpl.replace(b'return X;', b'return ' + str(v).encode() + b';')
    j = b.find(cand, i)
    if j >= 0:
        found = (j, cand, v); break
if found is None:
    print('NO_CASE_BLOCK'); sys.exit(3)
j, cand, v = found
print('CURRENT_NWARPS=' + str(v))
if v == NW:
    print('ALREADY_AT_TARGET'); print('MD5_AFTER=' + md5()); sys.exit(0)
new = cand.replace(b'return ' + str(v).encode() + b';', b'return ' + str(NW).encode() + b';')
F.write_bytes(b[:j] + new + b[j+len(cand):])
print('MD5_AFTER=' + md5())
print('PATCHED_TO=' + str(NW))
