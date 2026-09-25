#include "common.cuh"
#include "convert.cuh"
#include "fattn-common.cuh"

// R362: route SM70 D256 decode attention (q 1..8) to the vendored 1cat kernel
// (grouped-attention.cu, built into ggml-cuda-sm70long). Off by default; the
// stock mma_f16 path handles decode unless LLAMA_SM70_LONG_DECODE=1.

extern "C" void sm70_long_decode_f16(
    const void * q, const void * k_cache, const void * v_cache, void * out,
    const void * block_table, const void * seq_lens,
    void * partial, void * max_logits, void * exp_sums, void * online_rescales,
    const void * active_num_partitions,
    int n_q, int n_kv_heads, int page_size, int n_pages, int n_parts,
    int n_q_heads_per_kv, float softmax_scale, cudaStream_t stream);

static constexpr int SM70_LONG_PAGE_TOKENS = 256;

static bool sm70_long_decode_enabled() {
    static const bool enabled = getenv("LLAMA_SM70_LONG_DECODE") != nullptr &&
                                atoi(getenv("LLAMA_SM70_LONG_DECODE")) != 0;
    return enabled;
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
    if (Q->ne[0] != 256 || V->ne[0] != 256) {
        return false;
    }
    if (Q->ne[1] < 1 || Q->ne[1] > 8) {
        return false; // decode only
    }
    if (K->ne[1] < SM70_LONG_PAGE_TOKENS * 2) {
        return false; // long context only
    }
    if (Q->ne[2] % K->ne[2] != 0 || Q->ne[2] / K->ne[2] != 6) {
        return false; // the vendored kernel is instantiated for GQA 6:1
    }
    if (K->ne[3] != 1 || V->ne[3] != 1 || Q->ne[3] != 1) {
        return false;
    }
    if (K->type != V->type) {
        return false;
    }
    // The vendored kernel reads each token's head dimension contiguously, so the
    // K/V mirror must be buildable: accept the packed layouts to_fp16 handles.
    if (!ggml_is_contiguous(Q)) {
        return false;
    }
    return true;
}

void ggml_cuda_sm70_long_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const int64_t n_q      = Q->ne[1];
    const int64_t hq       = Q->ne[2];
    const int64_t hkv      = K->ne[2];
    const int64_t kv_len   = K->ne[1];
    const int64_t gqa      = hq / hkv;
    const int64_t n_pages  = (kv_len + SM70_LONG_PAGE_TOKENS - 1) / SM70_LONG_PAGE_TOKENS;

    // one partition per 256 KV tokens, capped to keep the scratch bounded
    int n_parts = (int) ((kv_len + 255) / 256);
    if (n_parts > 512) {
        n_parts = 512;
    }
    if (n_parts < 1) {
        n_parts = 1;
    }

    cudaStream_t stream = ctx.stream();
    const int device = ctx.device;
    GGML_ASSERT(ggml_backend_cuda_get_device_count() > 0);
    cudaSetDevice(device);

    // ------------------------------------------------------------------ KV mirror
    const size_t k_elems = (size_t) 256 * kv_len * hkv;
    half * K_f16 = nullptr;
    half * V_f16 = nullptr;
    CUDA_CHECK(cudaMalloc(&K_f16, k_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&V_f16, k_elems * sizeof(half)));

    {
        const int64_t ne = ggml_nelements(K);
        if (K->type == GGML_TYPE_F16) {
            CUDA_CHECK(cudaMemcpyAsync(K_f16, K->data, k_elems * sizeof(half),
                                       cudaMemcpyDeviceToDevice, stream));
        } else {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            GGML_ASSERT(to_fp16 != nullptr);
            to_fp16(K->data, K_f16, ne, stream);
        }
        if (V->type == GGML_TYPE_F16) {
            CUDA_CHECK(cudaMemcpyAsync(V_f16, V->data, k_elems * sizeof(half),
                                       cudaMemcpyDeviceToDevice, stream));
        } else {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
            GGML_ASSERT(to_fp16 != nullptr);
            to_fp16(V->data, V_f16, ne, stream);
        }
    }

    // Q must be f16 and contiguous: [n_q, hq, 256]
    half * Q_f16 = nullptr;
    CUDA_CHECK(cudaMalloc(&Q_f16, (size_t) 256 * n_q * hq * sizeof(half)));
    {
        to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(Q->type);
        GGML_ASSERT(to_fp16 != nullptr);
        if (ggml_is_contiguous(Q) && Q->type == GGML_TYPE_F16) {
            CUDA_CHECK(cudaMemcpyAsync(Q_f16, Q->data, (size_t) 256 * n_q * hq * sizeof(half),
                                       cudaMemcpyDeviceToDevice, stream));
        } else {
            to_fp16(Q->data, Q_f16, ggml_nelements(Q), stream);
        }
    }

    // ------------------------------------------------------------------- scratch
    int * bt    = nullptr; // identity page table [1, n_pages]
    int * sl    = nullptr; // [1] visible length
    int * act   = nullptr; // [1] active partitions
    float * mxl = nullptr;
    float * exs = nullptr;
    float * ors = nullptr;
    half * part = nullptr;

    CUDA_CHECK(cudaMalloc(&bt,  n_pages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sl,  sizeof(int)));
    CUDA_CHECK(cudaMalloc(&act, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&mxl, (size_t) n_parts * n_q * gqa * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&exs, (size_t) n_parts * n_q * gqa * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ors, (size_t) n_q * gqa * n_parts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&part, (size_t) n_parts * n_q * gqa * 256 * sizeof(half)));

    {
        std::vector<int> h_bt(n_pages);
        for (int64_t i = 0; i < n_pages; ++i) {
            h_bt[i] = (int) i;
        }
        CUDA_CHECK(cudaMemcpyAsync(bt, h_bt.data(), n_pages * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        const int h_sl = (int) kv_len;
        const int h_act = n_parts;
        CUDA_CHECK(cudaMemcpyAsync(sl, &h_sl, sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(act, &h_act, sizeof(int), cudaMemcpyHostToDevice, stream));
    }

    // output: f16 buffer then convert back to the dst type
    half * out_f16 = nullptr;
    CUDA_CHECK(cudaMalloc(&out_f16, (size_t) 256 * n_q * hq * sizeof(half)));

    const float scale = 1.0f / sqrtf((float) 256);

    // the vendored kernel is per KV head: launch once per local KV head
    for (int64_t h = 0; h < hkv; ++h) {
        sm70_long_decode_f16(
            (const void *) ((const char *) Q_f16 + (size_t) h * gqa * 256 * sizeof(half)),
            (const void *) K_f16,
            (const void *) V_f16,
            (void *) ((char *) out_f16 + (size_t) h * gqa * 256 * sizeof(half)),
            (const void *) bt, (const void *) sl,
            (void *) part, (void *) mxl, (void *) exs, (void *) ors,
            (const void *) act,
            (int) n_q, /*n_kv_heads=*/1, SM70_LONG_PAGE_TOKENS, (int) n_pages, n_parts,
            (int) gqa, scale, stream);
    }

    // convert the f16 output back to the destination type
    if (dst->type == GGML_TYPE_F16) {
        CUDA_CHECK(cudaMemcpyAsync(dst->data, out_f16, (size_t) 256 * n_q * hq * sizeof(half),
                                   cudaMemcpyDeviceToDevice, stream));
    } else {
        to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        GGML_ASSERT(to_fp32 != nullptr);
        to_fp32(out_f16, (float *) dst->data, 256 * n_q * hq, stream);
    }

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

size_t ggml_cuda_sm70_long_decode_alloc_size(const ggml_tensor * dst) {
    // scratch is allocated with cudaMalloc inside the launch; the scheduler only
    // needs room for the destination itself.
    GGML_UNUSED(dst);
    return sizeof(float) * ggml_nelements(dst);
}
