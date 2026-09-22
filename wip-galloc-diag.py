#!/usr/bin/env python3
# R251: diagnose WHY ggml_gallocr_alloc_graph fails (which tensors grow / node count changes).
# usage: patch-galloc.py <ggml/src dir> [revert]
import hashlib, sys, pathlib
ROOT = pathlib.Path(sys.argv[1]); MODE = sys.argv[2] if len(sys.argv) > 2 else 'apply'
F = ROOT / 'ggml-alloc.c'
BSN = bytes([92]) + b'n'
def md5(): return hashlib.md5(F.read_bytes()).hexdigest()
b = F.read_bytes()
EOL = b'\r\n' if b'\r\n' in b[:4000] else b'\n'
print('MD5_BEFORE=' + md5())

DECL = EOL.join([
  b'    const bool diag = (getenv("GGML_GALLOCR_DIAG") != NULL);',
  b'    static int64_t d_calls = 0, d_nodes = 0, d_leafs = 0, d_grow = 0;',
  b'    static char d_last[256] = {0};',
  b'    d_calls++;',
  b'    if (diag && (d_calls % 128 == 0)) {',
  b'        fprintf(stderr, "[GALLOC] calls=%lld nodes_diff=%lld leafs_diff=%lld grow=%lld | last=%s' + BSN + b'",',
  b'                (long long) d_calls, (long long) d_nodes, (long long) d_leafs, (long long) d_grow, d_last);',
  b'    }'])

A1 = b'static bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {'
N1 = A1 + EOL + DECL

A2 = b'    if (galloc->n_nodes != graph->n_nodes) {'
N2 = A2 + EOL + b'        d_nodes++; snprintf(d_last, sizeof(d_last), "nodes %d->%d", galloc->n_nodes, graph->n_nodes);'

A3 = b'    if (galloc->n_leafs != graph->n_leafs) {'
N3 = A3 + EOL + b'        d_leafs++; snprintf(d_last, sizeof(d_last), "leafs %d->%d", galloc->n_leafs, graph->n_leafs);'

A4 = b'        if (!ggml_gallocr_node_needs_realloc(galloc, node, &node_alloc->dst)) {'
N4 = A4 + EOL + b'            d_grow++; snprintf(d_last, sizeof(d_last), "grow dst %s", node->name);'

A5 = b'            if (!ggml_gallocr_node_needs_realloc(galloc, src, &node_alloc->src[j])) {'
N5 = A5 + EOL + b'                d_grow++; snprintf(d_last, sizeof(d_last), "grow src %s of %s", src->name, node->name);'

pairs = [(A1, N1), (A2, N2), (A3, N3), (A4, N4), (A5, N5)]
if MODE == 'revert':
    pairs = [(n, a) for (a, n) in pairs]
for k, (old, new) in enumerate(pairs):
    n = b.count(old)
    print('HITS_' + str(k) + '=' + str(n))
    if n != 1:
        print('ANCHOR_FAIL'); sys.exit(2)
b2 = b
for (old, new) in pairs:
    b2 = b2.replace(old, new, 1)
F.write_bytes(b2)
print('LINES_DELTA=' + str(b2.count(EOL) - b.count(EOL)))
print('MD5_AFTER=' + md5())
print('DONE_' + MODE)
