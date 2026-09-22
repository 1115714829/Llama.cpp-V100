#!/usr/bin/env python3
# R250c: count gallocr slow-path reallocations (diagnostic, server tree only). usage: patch-realloc.py <ggml/src dir> [revert]
import hashlib, sys, pathlib
ROOT = pathlib.Path(sys.argv[1]); MODE = sys.argv[2] if len(sys.argv) > 2 else 'apply'
F = ROOT / 'ggml-backend.cpp'
BSN = bytes([92]) + b'n'
def md5(): return hashlib.md5(F.read_bytes()).hexdigest()
b = F.read_bytes()
EOL = b'\r\n' if b'\r\n' in b[:4000] else b'\n'
print('MD5_BEFORE=' + md5())
A = EOL.join([b'        if (sched->debug_realloc > 0) {'])
N = EOL.join([
  b'        {',
  b'            static int64_t n_realloc_total = 0;',
  b'            static int64_t n_realloc_unexpected = 0;',
  b'            static const bool rc_report = (getenv("GGML_SCHED_REALLOC_COUNT") != nullptr);',
  b'            n_realloc_total++;',
  b'            if (!backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size) {',
  b'                n_realloc_unexpected++;',
  b'            }',
  b'            if (rc_report && (n_realloc_total % 16 == 0)) {',
  b'                fprintf(stderr, "[REALLOC] total=%lld unexpected=%lld size=%d nodes=%d leafs=%d' + BSN + b'",',
  b'                        (long long) n_realloc_total, (long long) n_realloc_unexpected,',
  b'                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs);',
  b'            }',
  b'        }',
  b'        if (sched->debug_realloc > 0) {'])
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
