#include "common.cuh"
#include "fattn-common.cuh"

#include <chrono>

// R363: route SM70 D256 decode attention (q 1..8) to the vendored 1cat kernel
// (grouped-attention.cu, built into ggml-cuda-sm70long). Off by default; the
// stock mma_f16 path handles decode unless LLAMA_SM70_LONG_DECODE=1.
//
// Per-device shapes measured on the 4-card TP4 service:
//   Q ne=(256,1,24,1) nb1=24576 nb2=1024  -> [t][h][d] with the head dim fastest
//   K ne=(256,262144,4,1) nb1=1088 nb2=272 -> position-major [t][hkv][d], q8_0 rows
// so each device owns all 24 Q heads and all 4 KV heads, and the kernel is
// launched once per KV head with that head's 6 Q heads.

extern "C" void sm70_long_decode_f16(
    const void * q, const void * k_cache, const void * v_cache, void * out,
    const void * block_table, const void * seq_lens,
    void * partial, void * max_logits, void * exp_sums, void * online_rescales,
    const void * active_num_partitions,
    int n_q, int n_kv_heads, int page_size, int n_pages, int n_parts,
    int n_q_heads_per_kv, float softmax_scale, cudaStream_t stream);

// R410: E4M3 KV cache, read in place (no staging). Same shapes, 1 byte per value.
extern "C" void sm70_long_decode_fp8(
    const void * q, const void * k_cache, const void * v_cache, void * out,
    const void * block_table, const void * row_lengths,
    void * partial, void * lse,
    int q_rows, int n_kv_heads, int n_q_heads_per_kv,
    int page_tokens, int n_pages,
    float k_scale, float v_scale, float softmax_scale, cudaStream_t stream);

static constexpr int SM70_LONG_PAGE_TOKENS = 256;
static constexpr int SM70_LONG_D           = 256;
// R367: the vendored grouped verifier's compile-time workspace
// (kGroupedVerifyQ8Splits x kGroupedVerifyQ8MaxQ x 6 heads), see grouped-attention.cu.
static constexpr int SM70_LONG_WS_SPLITS   = 80;
static constexpr int SM70_LONG_WS_MAXQ     = 8;

static bool sm70_long_decode_enabled() {
    static const bool enabled = getenv("LLAMA_SM70_LONG_DECODE") != nullptr &&
                                atoi(getenv("LLAMA_SM70_LONG_DECODE")) != 0;
    return enabled;
}

// diagnostic (R365): report the outcome for decode-shaped calls that already carry a
// long KV (>= 512), which is the case this kernel targets. Short-KV calls stay silent
// so the print budget is not consumed by the early small-context steps.
static void sm70_long_note(const char * what, const ggml_tensor * Q, const ggml_tensor * K) {
    static int printed = 0;
    if (K->ne[1] < SM70_LONG_PAGE_TOKENS * 2) {
        return;
    }
    if (printed < 40) {
        printed++;
        fprintf(stderr, "[SM70LONG] #%d %s | q=%lld kv=%lld hq=%lld hkv=%lld Ktype=%d Qtype=%d nb1=%zu nb2=%zu\n",
                printed, what,
                (long long) Q->ne[1], (long long) K->ne[1],
                (long long) Q->ne[2], (long long) K->ne[2],
                (int) K->type, (int) Q->type, K->nb[1], K->nb[2]);
    }
}

// Q staging: src [t][hq][d] (f32 or f16) -> dst [gqa][t][d] f16 for KV group g.
static __global__ void sm70_long_stage_q(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t n_q, const int64_t gqa, const int64_t g,
        const size_t nb1, const size_t nb2, const bool src_f16) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = n_q * gqa * SM70_LONG_D;
    if (i >= total) {
        return;
    }
    const int     d  = (int) (i % SM70_LONG_D);
    const int64_t hh = (i / SM70_LONG_D) % gqa;
    const int64_t t  = i / (SM70_LONG_D * gqa);
    const char * p = src + (size_t) t * nb1 + (size_t) (g * gqa + hh) * nb2;
    const float v = src_f16 ? __half2float(*(const half *) p)
                            : *(const float *) p;
    dst[i] = __float2half(v);
}

// K/V staging: rows of 256 values at (t, j) -> token-major [t][256] f16.
// q8_0 rows are 8 blocks of 34 bytes; f16 rows are 256 halves.
static __global__ void sm70_long_stage_kv(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t n_tokens, const int64_t j,
        const size_t nb1, const size_t nb2, const bool q8_0) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = n_tokens * SM70_LONG_D;
    if (i >= total) {
        return;
    }
    const int     d = (int) (i % SM70_LONG_D);
    const int64_t t = i / SM70_LONG_D;
    const char * row = src + (size_t) t * nb1 + (size_t) j * nb2;
    float v;
    if (q8_0) {
        const char * blk = row + (d / 32) * 34;
        const float scale = __half2float(*(const __half *) blk);
        v = scale * (float) (signed char) blk[2 + (d % 32)];
    } else {
        v = __half2float(((const half *) row)[d]);
    }
    dst[i] = __float2half(v);
}

// Output scatter: compact per-group [gqa][t][d] f16 -> dst [t][hq][d].
static __global__ void sm70_long_scatter_out(
        const half * __restrict__ src, char * __restrict__ dst,
        const int64_t n_q, const int64_t gqa, const int64_t g,
        const size_t nb1, const size_t nb2, const bool dst_f16) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = n_q * gqa * SM70_LONG_D;
    if (i >= total) {
        return;
    }
    const int     d  = (int) (i % SM70_LONG_D);
    const int64_t hh = (i / SM70_LONG_D) % gqa;
    const int64_t t  = i / (SM70_LONG_D * gqa);
    char * p = dst + (size_t) t * nb1 + (size_t) (g * gqa + hh) * nb2;
    if (dst_f16) {
        *(half *) p = src[i];
    } else {
        *(float *) p = __half2float(src[i]);
    }
}

// R368: persistent per-device workspace. Per-call cudaMalloc/cudaFree is illegal
// inside the meta backend's graph capture and costs hundreds of allocs per step.
struct sm70_long_ws {
    half * k = nullptr;
    half * v = nullptr;
    half * q = nullptr;
    half * o = nullptr;
    int  * bt = nullptr;
    int  * sl = nullptr;
    int  * act = nullptr;
    float * mxl = nullptr;
    float * exs = nullptr;
    float * ors = nullptr;
    half  * part = nullptr;
    // R410: fp8 route needs float partials and a PAIRED lse (lse, sum) buffer; both are
    // fixed sizes so they are allocated once and never resized.
    float * partf = nullptr;   // [80][8][6][256]
    float * lsef  = nullptr;   // [80][8][6][2]
    size_t kv_cap = 0;    // elements per K/V plane (D * kv_len)
    size_t q_cap  = 0;    // elements per Q/O buffer (D * n_q_pad * hq)
    size_t bt_cap = 0;    // ints (n_q_pad * n_pages)
    bool   stats_ok = false;
};

static sm70_long_ws & sm70_long_ws_for(int dev) {
    static sm70_long_ws ws[GGML_CUDA_MAX_DEVICES];
    return ws[dev >= 0 && dev < GGML_CUDA_MAX_DEVICES ? dev : 0];
}

bool ggml_cuda_sm70_long_decode_supported(int cc, const ggml_tensor * dst) {
    if (!sm70_long_decode_enabled()) {
        return false;
    }
    if (cc != GGML_CUDA_CC_VOLTA) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (Q == nullptr || K == nullptr || V == nullptr || mask == nullptr) {
        return false;
    }
    if (Q->ne[0] != SM70_LONG_D || V->ne[0] != SM70_LONG_D) {
        return false;
    }
    if (Q->ne[1] < 1 || Q->ne[1] > 8) {
        return false; // decode only (prefill is silent: it always fails here)
    }
    if (K->ne[1] < 1) {
        sm70_long_note("empty kv", Q, K);
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0 || Q->ne[2] / K->ne[2] != 6) {
        sm70_long_note("GQA ratio is not 6:1", Q, K);
        return false; // the vendored kernel is instantiated for GQA 6:1
    }
    if (K->ne[3] != 1 || V->ne[3] != 1 || Q->ne[3] != 1) {
        sm70_long_note("ne3 != 1", Q, K);
        return false;
    }
    if (K->type != V->type) {
        sm70_long_note("K/V type mismatch", Q, K);
        return false;
    }
    if (!(K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_F8_E4M3)) {
        sm70_long_note("K/V type is neither f16 nor q8_0", Q, K);
        return false;
    }
    if (Q->type != GGML_TYPE_F32 && Q->type != GGML_TYPE_F16) {
        sm70_long_note("Q type is neither f32 nor f16", Q, K);
        return false;
    }
    // R367: the runtime node is head-split under TP (6 Q / 1 KV per device) while the
    // reserve graph carries the pre-split model shape (24/4). Accept any GQA 6:1 pair.
    if (K->ne[2] < 1 || K->ne[2] > 4 || Q->ne[2] != K->ne[2] * 6) {
        sm70_long_note("head pair is not GQA 6:1", Q, K);
        return false;
    }
    sm70_long_note("SELECTED (all checks passed)", Q, K);
    return true;
}

size_t ggml_cuda_sm70_long_decode_alloc_size(const ggml_tensor * dst) {
    GGML_UNUSED(dst);
    // scratch is allocated with cudaMalloc inside the launch
    return sizeof(float) * ggml_nelements(dst);
}

void ggml_cuda_sm70_long_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // R366: ground truth for "does this kernel run at all", plus an optional cost meter.
    // cudaFree at the end of the body makes the host timing cover the whole path.
    static const bool prof = getenv("LLAMA_SM70_LONG_PROF") != nullptr;
    static double acc_ms = 0.0;
    static long   acc_n  = 0;
    const auto    t0     = std::chrono::steady_clock::now();
    {
        static int printed = 0;
        if (printed < 4) {
            printed++;
            fprintf(stderr, "[SM70EXEC] #%d dev=%d q=%lld kv=%lld hq=%lld hkv=%lld Ktype=%d dsttype=%d\n",
                    printed, ctx.device,
                    (long long) Q->ne[1], (long long) K->ne[1],
                    (long long) Q->ne[2], (long long) K->ne[2],
                    (int) K->type, (int) dst->type);
        }
    }

    const int64_t n_q    = Q->ne[1];
    const int64_t hq     = Q->ne[2];
    const int64_t hkv    = K->ne[2];
    const int64_t kv_len = K->ne[1];
    const int64_t gqa    = hq / hkv;
    const int64_t n_pages = (kv_len + SM70_LONG_PAGE_TOKENS - 1) / SM70_LONG_PAGE_TOKENS;

    // R367: the vendored kernel takes one partition per 256-token page. The old 512 cap
    // silently truncated long contexts (256K needs 1024 pages) and caused an IMA.
    int n_parts = (int) ((kv_len + SM70_LONG_PAGE_TOKENS - 1) / SM70_LONG_PAGE_TOKENS);
    if (n_parts < 1) {
        n_parts = 1;
    }

    cudaStream_t stream = ctx.stream();
    cudaSetDevice(ctx.device);

    const size_t kv_elems = (size_t) SM70_LONG_D * kv_len; // per head

    // R368: persistent per-device workspace. Per-call cudaMalloc/cudaFree is illegal
    // inside the meta backend's graph capture and costs hundreds of allocs per step.
    sm70_long_ws & ws = sm70_long_ws_for(ctx.device);
    const int64_t n_q_pad = n_q < 2 ? 2 : n_q;   // the wrapper pads q to >= 2 rows
    const size_t kv_need = kv_elems * hkv;
    const size_t q_need  = (size_t) SM70_LONG_D * n_q_pad * hq;
    if (ws.kv_cap < kv_need) {
        if (ws.k != nullptr) { CUDA_CHECK(cudaFree(ws.k)); ws.k = nullptr; }
        if (ws.v != nullptr) { CUDA_CHECK(cudaFree(ws.v)); ws.v = nullptr; }
        CUDA_CHECK(cudaMalloc(&ws.k, kv_need * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&ws.v, kv_need * sizeof(half)));
        ws.kv_cap = kv_need;
    }
    if (ws.q_cap < q_need) {
        if (ws.q != nullptr) { CUDA_CHECK(cudaFree(ws.q)); ws.q = nullptr; }
        if (ws.o != nullptr) { CUDA_CHECK(cudaFree(ws.o)); ws.o = nullptr; }
        CUDA_CHECK(cudaMalloc(&ws.q, q_need * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&ws.o, q_need * sizeof(half)));
        ws.q_cap = q_need;
    }
    half * K_f16 = ws.k;
    half * V_f16 = ws.v;
    half * Q_f16 = ws.q;
    half * out_f16 = ws.o;
    CUDA_CHECK(cudaMemsetAsync(Q_f16, 0, q_need * sizeof(half), stream));
    CUDA_CHECK(cudaMemsetAsync(out_f16, 0, q_need * sizeof(half), stream));

    const bool k_q8 = K->type == GGML_TYPE_Q8_0;
    const bool v_q8 = V->type == GGML_TYPE_Q8_0;
    const bool q_f16 = Q->type == GGML_TYPE_F16;
    const int  threads = 256;

    // R368: an f16 KV cache in token-major [t][d] order is already the kernel's paged
    // layout (identity block table), so pass it through without a staging copy.
    const bool k_direct = !k_q8 && hkv == 1 && K->nb[1] == SM70_LONG_D * sizeof(half);
    const bool v_direct = !v_q8 && hkv == 1 && V->nb[1] == SM70_LONG_D * sizeof(half);
    if (k_direct) { K_f16 = (half *) K->data; }
    if (v_direct) { V_f16 = (half *) V->data; }

    // R410: E4M3 KV is read in place by the grouped-verify kernel, so no KV staging.
    if ((!k_direct || !v_direct) && K->type != GGML_TYPE_F8_E4M3) {
        for (int64_t j = 0; j < hkv; ++j) {
            if (!k_direct) {
                sm70_long_stage_kv<<<(int) ((kv_elems + threads - 1) / threads), threads, 0, stream>>>(
                    (const char *) K->data, ws.k + (size_t) j * kv_elems,
                    kv_len, j, K->nb[1], K->nb[2], k_q8);
            }
            if (!v_direct) {
                sm70_long_stage_kv<<<(int) ((kv_elems + threads - 1) / threads), threads, 0, stream>>>(
                    (const char *) V->data, ws.v + (size_t) j * kv_elems,
                    kv_len, j, V->nb[1], V->nb[2], v_q8);
            }
        }
    }
    {
        // compact per-group Q: group j occupies a [n_q][gqa][256] plane, plane stride n_q_pad
        const int64_t total_g = (int64_t) SM70_LONG_D * n_q_pad * gqa;
        for (int64_t j = 0; j < hkv; ++j) {
            sm70_long_stage_q<<<(int) ((total_g + threads - 1) / threads), threads, 0, stream>>>(
                (const char *) Q->data, Q_f16 + (size_t) j * total_g,
                n_q, gqa, j, Q->nb[1], Q->nb[2], q_f16);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    // R368: scratch lives in the persistent workspace (see sm70_long_ws).
    const size_t bt_need  = (size_t) n_q_pad * n_pages;
    const size_t stat_need = (size_t) SM70_LONG_WS_SPLITS * SM70_LONG_WS_MAXQ * gqa;
    if (ws.bt_cap < bt_need) {
        if (ws.bt != nullptr) { CUDA_CHECK(cudaFree(ws.bt)); ws.bt = nullptr; }
        CUDA_CHECK(cudaMalloc(&ws.bt, bt_need * sizeof(int)));
        ws.bt_cap = bt_need;
    }
    if (!ws.stats_ok) {
        CUDA_CHECK(cudaMalloc(&ws.sl,  (size_t) n_q_pad * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws.act, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws.mxl, stat_need * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws.exs, stat_need * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws.ors, stat_need * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws.part, stat_need * SM70_LONG_D * sizeof(half)));
        ws.stats_ok = true;
    }
    int * bt  = ws.bt;
    int * sl  = ws.sl;
    int * act = ws.act;
    float * mxl = ws.mxl;
    float * exs = ws.exs;
    float * ors = ws.ors;
    half  * part = ws.part;

    {
        // R367: the kernel indexes block_table/seq_lens by query token (batch_idx = q row),
        // so both need one row per query token, not one row per request.
        std::vector<int> h_bt(bt_need);
        for (int64_t r = 0; r < n_q_pad; ++r) {
            for (int64_t i = 0; i < n_pages; ++i) {
                h_bt[(size_t) r * n_pages + i] = (int) i;
            }
        }
        std::vector<int> h_sl((size_t) n_q_pad, (int) kv_len);
        const int h_act = n_parts < SM70_LONG_WS_SPLITS ? n_parts : SM70_LONG_WS_SPLITS;
        CUDA_CHECK(cudaMemcpyAsync(bt, h_bt.data(), h_bt.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(sl, h_sl.data(), h_sl.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(act, &h_act, sizeof(int), cudaMemcpyHostToDevice, stream));
    }

    const float scale = 1.0f / sqrtf((float) SM70_LONG_D);

    if (K->type == GGML_TYPE_F8_E4M3) {
        // R410: the vendored grouped-verify kernel reads the flat E4M3 cache in place.
        // It needs float partials and a paired (lse, sum) buffer; both have fixed size.
        if (ws.partf == nullptr) {
            CUDA_CHECK(cudaMalloc(&ws.partf, (size_t) SM70_LONG_WS_SPLITS * SM70_LONG_WS_MAXQ * gqa * SM70_LONG_D * sizeof(float)));
        }
        if (ws.lsef == nullptr) {
            CUDA_CHECK(cudaMalloc(&ws.lsef, (size_t) SM70_LONG_WS_SPLITS * SM70_LONG_WS_MAXQ * gqa * 2 * sizeof(float)));
        }
        const int64_t total_g = (int64_t) SM70_LONG_D * n_q_pad * gqa;
        const int64_t kv_head_off = 256; // one head of a token row is 256 E4M3 bytes
        for (int64_t j = 0; j < hkv; ++j) {
            sm70_long_decode_fp8(
                (const void *) (Q_f16 + (size_t) j * total_g),
                (const void *) ((const char *) K->data + j * kv_head_off),
                (const void *) ((const char *) V->data + j * kv_head_off),
                (void *)       (out_f16 + (size_t) j * total_g),
                (const void *) bt, (const void *) sl,
                (void *) ws.partf, (void *) ws.lsef,
                (int) n_q, (int) hkv, (int) gqa, SM70_LONG_PAGE_TOKENS, (int) n_pages,
                1.0f, 1.0f, scale, stream);
        }
        for (int64_t j = 0; j < hkv; ++j) {
            sm70_long_scatter_out<<<(int) ((total_g + threads - 1) / threads), threads, 0, stream>>>(
                out_f16 + (size_t) j * total_g, (char *) dst->data,
                n_q, gqa, j, dst->nb[1], dst->nb[2], dst->type == GGML_TYPE_F16);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // one launch per KV head, with that head's gqa Q heads
    {
        const int64_t total_g = (int64_t) SM70_LONG_D * n_q_pad * gqa;
        for (int64_t j = 0; j < hkv; ++j) {
            sm70_long_decode_f16(
                (const void *) (Q_f16 + (size_t) j * total_g), // [n_q][gqa][256] plane per KV head
                (const void *) (K_f16 + (size_t) j * kv_elems),
                (const void *) (V_f16 + (size_t) j * kv_elems),
                (void *)       (out_f16 + (size_t) j * total_g),
                (const void *) bt, (const void *) sl,
                (void *) part, (void *) mxl, (void *) exs, (void *) ors,
                (const void *) act,
                (int) n_q, /*n_kv_heads=*/1, SM70_LONG_PAGE_TOKENS, (int) n_pages, n_parts,
                (int) gqa, scale, stream);
        }
    }

    // scatter the compact per-group outputs back into dst [t][hq][d]
    {
        const int64_t total_g = (int64_t) SM70_LONG_D * n_q_pad * gqa;
        const bool dst_f16 = dst->type == GGML_TYPE_F16;
        for (int64_t j = 0; j < hkv; ++j) {
            sm70_long_scatter_out<<<(int) ((total_g + threads - 1) / threads), threads, 0, stream>>>(
                out_f16 + (size_t) j * total_g, (char *) dst->data,
                n_q, gqa, j, dst->nb[1], dst->nb[2], dst_f16);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    // R368: the workspace is persistent; freeing per call is illegal under graph capture.

    if (prof) {
        acc_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        acc_n++;
        if (acc_n % 64 == 0) {
            fprintf(stderr, "[SM70PROF] calls=%ld avg=%.3f ms\n", acc_n, acc_ms / (double) acc_n);
        }
    }
}
