#!/usr/bin/env python3
# R256: make the B3 vectorised selector the DEFAULT (env now only used to force the scalar path with =0).
# usage: patch-seldef.py <common dir> [revert]
import hashlib, sys, pathlib
ROOT = pathlib.Path(sys.argv[1]); MODE = sys.argv[2] if len(sys.argv) > 2 else 'apply'
F = ROOT / 'speculative.cpp'
def md5(): return hashlib.md5(F.read_bytes()).hexdigest()
b = F.read_bytes()
EOL = b'\r\n' if b'\r\n' in b[:4000] else b'\n'
print('MD5_BEFORE=' + md5())
A = b'        const bool sel_vec = getenv("GGML_SPEC_SELECTOR_VEC") != nullptr;'
N = EOL.join([
  b'        // R256: the lane-vectorised gate is the default now (it measures -0.33 ms per call and is',
  b'        // bit-identical); GGML_SPEC_SELECTOR_VEC=0 forces the scalar path back for A/B runs.',
  b'        const char * sel_vec_env = getenv("GGML_SPEC_SELECTOR_VEC");',
  b'        const bool sel_vec = (sel_vec_env == nullptr) || (atoi(sel_vec_env) != 0);'])
old, new = (A, N) if MODE == 'apply' else (N, A)
n = b.count(old)
print('ANCHOR_HITS=' + str(n))
if n != 1:
    print('ANCHOR_FAIL'); sys.exit(2)
b2 = b.replace(old, new, 1)
F.write_bytes(b2)
print('LINES_DELTA=' + str(b2.count(EOL) - b.count(EOL)))
print('MD5_AFTER=' + md5())
print('DONE_' + MODE)
