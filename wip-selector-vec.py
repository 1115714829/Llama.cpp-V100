#!/usr/bin/env python3
# R232: DFlash2 CPU selector -- vectorise the gate phase across block positions (bit-identical).
# bytes-only I/O, CRLF aware, anchor uniqueness enforced. usage: patch-selector.py [ROOT]
import hashlib, sys, pathlib
ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '/root/llm/test/v100-opt/llama.cpp/common')
F = ROOT / 'speculative.cpp'
NL = b'\r\n'
def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()
def edit(b, old, new):
    if new in b:
        print('ALREADY_APPLIED'); return b
    n = b.count(old)
    if n != 1:
        print('ANCHOR_MISS found=' + str(n)); sys.exit(2)
    return b.replace(old, new, 1)

b = F.read_bytes()
print('MD5_BEFORE=' + md5(F))

# 1) topk: fixed-size buffers instead of two std::vector allocations per position
a1 = NL.join([b'                std::vector<int32_t> ids(top_k, 0);',
              b'                std::vector<float>   vals(top_k, -INFINITY);'])
n1 = NL.join([b'                GGML_ASSERT(top_k <= 16);',
              b'                int32_t ids[16];',
              b'                float   vals[16];',
              b'                for (int32_t k = 0; k < top_k; ++k) { ids[k] = 0; vals[k] = -INFINITY; }'])
b = edit(b, a1, n1)

# 2) transpose the block embeddings so the positions are contiguous (the vector lanes)
a2 = b'        auto work_gate = [&](int32_t k_beg, int32_t k_end) {'
n2 = NL.join([
  b'        // R232 (env GGML_SPEC_SELECTOR_VEC): the n_tokens dot products of one selector row share',
  b'        // that row, so the block positions become the vector lanes. Each lane keeps its own',
  b'        // accumulator and its own accumulation order, so every dot product is bit-identical to',
  b'        // the scalar form while the row load is amortised over the lanes.',
  b'        const bool sel_vec = getenv("GGML_SPEC_SELECTOR_VEC") != nullptr;',
  b'        std::vector<float> embd_T;',
  b'        if (sel_vec) {',
  b'            embd_T.resize((size_t) n_embd_dec * n_tokens);',
  b'            for (int32_t i = 0; i < n_tokens; ++i) {',
  b'                const float * e = embd_all + (size_t) i * n_embd_dec;',
  b'                for (int32_t d = 0; d < n_embd_dec; ++d) {',
  b'                    embd_T[(size_t) d * n_tokens + i] = e[d];',
  b'                }',
  b'            }',
  b'        }',
  b'        auto work_gate = [&](int32_t k_beg, int32_t k_end) {'])
b = edit(b, a2, n2)

# 3) the vectorised lane body, inserted right after the row pointer is taken
a3 = b'                const float * row = sel_hidden.data() + (size_t) k * n_embd_dec;'
vec = []
vec.append(b'                if (sel_vec && n_tokens == 8) {')
vec.append(b'                    float s0[8] = {0}, s1[8] = {0}, s2[8] = {0}, s3[8] = {0};')
vec.append(b'                    int32_t d = 0;')
vec.append(b'                    for (; d + 4 <= n_embd_dec; d += 4) {')
vec.append(b'                        const float c0 = row[d + 0], c1 = row[d + 1], c2 = row[d + 2], c3 = row[d + 3];')
for lane in range(4):
    vec.append(('                        const float * t%d = embd_T.data() + (size_t) (d + %d) * 8;' % (lane, lane)).encode())
for lane in range(4):
    for i in range(8):
        vec.append(('                        s%d[%d] += c%d * t%d[%d];' % (lane, i, lane, lane, i)).encode())
vec.append(b'                    }')
vec.append(b'                    for (int32_t i = 0; i < 8; ++i) {')
vec.append(b'                        float s = (s0[i] + s1[i]) + (s2[i] + s3[i]);')
vec.append(b'                        for (int32_t dd = d; dd < n_embd_dec; ++dd) {')
vec.append(b'                            s += row[dd] * embd_T[(size_t) dd * 8 + i];')
vec.append(b'                        }')
vec.append(b'                        gate[(size_t) i * rank + k] = s;')
vec.append(b'                    }')
vec.append(b'                    continue;')
vec.append(b'                }')
b = edit(b, a3, a3 + NL + NL.join(vec))

F.write_bytes(b)
print('MD5_AFTER=' + md5(F))
print('APPLIED')