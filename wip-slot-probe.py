#!/usr/bin/env python3
# R246: extend the graph-slot probe (cap 24 -> 400, add n_tokens) on the SERVER tree only.
# usage: patch-slot.py <src dir> [revert]
import hashlib, sys, pathlib
ROOT = pathlib.Path(sys.argv[1]); MODE = sys.argv[2] if len(sys.argv) > 2 else 'apply'
F = ROOT / 'llama-context.cpp'
def md5(): return hashlib.md5(F.read_bytes()).hexdigest()
b = F.read_bytes()
EOL = b'\r\n' if b'\r\n' in b[:4000] else b'\n'
print('MD5_BEFORE=' + md5())
A1 = b'        if (slot_n < 24) {'
A2 = EOL.join([b'            fprintf(stderr, "[SLOT] ctx=%s call=%d n_outputs=%u gtype=%d slot=%d res=%p prev_active=%p hit=%d\\n",',
               b'                    model.name.c_str(), slot_n, n_outputs, (int) gtype, slot, (const void *) res, (const void *) gf_res_prev_active, hit);'])
N1 = b'        if (slot_n < 400) {'
N2 = EOL.join([b'            fprintf(stderr, "[SLOT] ctx=%s call=%d n_tokens=%u n_outputs=%u gtype=%d slot=%d res=%p prev_active=%p hit=%d\\n",',
               b'                    model.name.c_str(), slot_n, ubatch.n_tokens, n_outputs, (int) gtype, slot, (const void *) res, (const void *) gf_res_prev_active, hit);'])
old, new = (A1, N1) if MODE == 'apply' else (N1, A1)
old2, new2 = (A2, N2) if MODE == 'apply' else (N2, A2)
for name, src in (('A1', old), ('A2', old2)):
    n = b.count(src)
    print('HITS_' + name + '=' + str(n))
    if n != 1:
        print('ANCHOR_FAIL'); sys.exit(2)
b2 = b.replace(old, new, 1).replace(old2, new2, 1)
F.write_bytes(b2)
dl = b2.count(EOL) - b.count(EOL)
print('LINES_DELTA=' + str(dl))
print('MD5_AFTER=' + md5())
print('DONE_' + MODE)
