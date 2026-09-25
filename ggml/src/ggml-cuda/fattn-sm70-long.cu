#include "common.cuh"
#include "fattn-common.cuh"

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

static constexpr int SM70_LONG_PAGE_TOKENS = 256;
static constexpr int SM70_LONG_D           = 256;

static bool sm70_long_decode_enabled() {
    static const bool enabled = getenv("LLAMA_SM70_LONG_DECODE") != nullptr &&
                                atoi(getenv("LLAMA_SM70_LONG_DECODE")) != 0;
    return enabled;
}

// diagnostic (R364): report decode-shaped calls and why they were not selected
static void sm70_long_reject(const char * why, const ggml_tensor * Q, const ggml_tensor * K) {
    static int printed = 0;
    if (printed < 8) {
        printed++;
        fprintf(stderr, "[SM70LONG] #%d reject: %s | q=%lld kv=%lld hq=%lld hkv=%lld Ktype=%d Qtype=%d\n",
                printed, why,
                (long long) Q->ne[1], (long long) K->ne[1],
                (long long) Q->ne[2], (long long) K->ne[2],
                (int) K->type, (int) Q->type);
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
    if (K->ne[1] < SM70_LONG_PAGE_TOKENS * 2) {
        sm70_long_reject("kv shorter than 512", Q, K);
        return false; // long context only
    }
    if (Q->ne[2] % K->ne[2] != 0 || Q->ne[2] / K->ne[2] != 6) {
        sm70_long_reject("GQA ratio is not 6:1", Q, K);
        return false; // the vendored kernel is instantiated for GQA 6:1
    }
    if (K->ne[3] != 1 || V->ne[3] != 1 || Q->ne[3] != 1) {
        sm70_long_reject("ne3 != 1", Q, K);
        return false;
    }
    if (K->type != V->type) {
        sm70_long_reject("K/V type mismatch", Q, K);
        return false;
    }
    if (!(K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0)) {
        sm70_long_reject("K/V type is neither f16 nor q8_0", Q, K);
        return false;
    }
    if (Q->type != GGML_TYPE_F32 && Q->type != GGML_TYPE_F16) {
        sm70_long_reject("Q type is neither f32 nor f16", Q, K);
        return false;
    }
    if (Q->ne[2] != 24 || K->ne[2] != 4) {
        sm70_long_reject("head counts are not 24/4", Q, K);
        return false; // shapes measured for this model/service
    }
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

    const int64_t n_q    = Q->ne[1];
    const int64_t hq     = Q->ne[2];
    const int64_t hkv    = K->ne[2];
    const int64_t kv_len = K->ne[1];
    const int64_t gqa    = hq / hkv;
    const int64_t n_pages = (kv_len + SM70_LONG_PAGE_TOKENS - 1) / SM70_LONG_PAGE_TOKENS;

    int n_parts = (int) ((kv_len + 255) / 256);
    if (n_parts > 512) {
        n_parts = 512;
    }
    if (n_parts < 1) {
        n_parts = 1;
    }

    cudaStream_t stream = ctx.stream();
    cudaSetDevice(ctx.device);

    const size_t kv_elems = (size_t) SM70_LONG_D * kv_len; // per head

    half * K_f16 = nullptr;
    half * V_f16 = nullptr;
    half * Q_f16 = nullptr;
    half * out_f16 = nullptr;
    CUDA_CHECK(cudaMalloc(&K_f16,   kv_elems * hkv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&V_f16,   kv_elems * hkv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&Q_f16,   (size_t) SM70_LONG_D * n_q * hq * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&out_f16, (size_t) SM70_LONG_D * n_q * hq * sizeof(half)));

    const bool k_q8 = K->type == GGML_TYPE_Q8_0;
    const bool v_q8 = V->type == GGML_TYPE_Q8_0;
    const bool q_f16 = Q->type == GGML_TYPE_F16;
    const int  threads = 256;

    for (int64_t j = 0; j < hkv; ++j) {
        sm70_long_stage_kv<<<(int) ((kv_elems + threads - 1) / threads), threads, 0, stream>>>(
            (const char *) K->data, K_f16 + (size_t) j * kv_elems,
            kv_len, j, K->nb[1], K->nb[2], k_q8);
        sm70_long_stage_kv<<<(int) ((kv_elems + threads - 1) / threads), threads, 0, stream>>>(
            (const char *) V->data, V_f16 + (size_t) j * kv_elems,
            kv_len, j, V->nb[1], V->nb[2], v_q8);
    }
    {
        // compact per-group Q: group j occupies [n_q][gqa][256] with token stride gqa*256
        const int64_t total_g = (int64_t) SM70_LONG_D * n_q * gqa;
        for (int64_t j = 0; j < hkv; ++j) {
            sm70_long_stage_q<<<(int) ((total_g + threads - 1) / threads), threads, 0, stream>>>(
                (const char *) Q->data, Q_f16 + (size_t) j * total_g,
                n_q, gqa, j, Q->nb[1], Q->nb[2], q_f16);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    int * bt  = nullptr;
    int * sl  = nullptr;
    int * act = nullptr;
    float * mxl = nullptr;
    float * exs = nullptr;
    float * ors = nullptr;
    half  * part = nullptr;

    CUDA_CHECK(cudaMalloc(&bt,  n_pages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sl,  sizeof(int)));
    CUDA_CHECK(cudaMalloc(&act, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&mxl, (size_t) n_parts * n_q * gqa * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&exs, (size_t) n_parts * n_q * gqa * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ors, (size_t) n_q * gqa * n_parts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&part, (size_t) n_parts * n_q * gqa * SM70_LONG_D * sizeof(half)));

    {
        std::vector<int> h_bt(n_pages);
        for (int64_t i = 0; i < n_pages; ++i) {
            h_bt[i] = (int) i;
        }
        const int h_sl = (int) kv_len;
        const int h_act = n_parts;
        CUDA_CHECK(cudaMemcpyAsync(bt, h_bt.data(), n_pages * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(sl, &h_sl, sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(act, &h_act, sizeof(int), cudaMemcpyHostToDevice, stream));
    }

    const float scale = 1.0f / sqrtf((float) SM70_LONG_D);

    // one launch per KV head, with that head's gqa Q heads
    for (int64_t j = 0; j < hkv; ++j) {
        sm70_long_decode_f16(
            (const void *) (Q_f16 + (size_t) j * gqa * SM70_LONG_D), // [n_q][gqa][256] view
            (const void *) (K_f16 + (size_t) j * kv_elems),
            (const void *) (V_f16 + (size_t) j * kv_elems),
            (void *)       (out_f16 + (size_t) j * gqa * SM70_LONG_D),
            (const void *) bt, (const void *) sl,
            (void *) part, (void *) mxl, (void *) exs, (void *) ors,
            (const void *) act,
            (int) n_q, /*n_kv_heads=*/1, SM70_LONG_PAGE_TOKENS, (int) n_pages, n_parts,
            (int) gqa, scale, stream);
    }

    // scatter the compact per-group outputs back into dst [t][hq][d]
    {
        const int64_t total_g = (int64_t) SM70_LONG_D * n_q * gqa;
        const bool dst_f16 = dst->type == GGML_TYPE_F16;
        for (int64_t j = 0; j < hkv; ++j) {
            sm70_long_scatter_out<<<(int) ((total_g + threads - 1) / threads), threads, 0, stream>>>(
                out_f16 + (size_t) j * total_g, (char *) dst->data,
                n_q, gqa, j, dst->nb[1], dst->nb[2], dst_f16);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(K_f16));
    CUDA_CHECK(cudaFree(V_f16));
    CUDA_CHECK(cudaFree(Q_f16));
    CUDA_CHECK(cudaFree(out_f16));
    CUDA_CHECK(cudaFree(bt));
    CUDA_CHECK(cudaFree(sl));
    CUDA_CHECK(cudaFree(act));
    CUDA_CHECK(cudaFree(mxl));
    CUDA_CHECK(cudaFree(exs));
    CUDA_CHECK(cudaFree(ors));
    CUDA_CHECK(cudaFree(part));
}
