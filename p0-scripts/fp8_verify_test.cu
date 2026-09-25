// Numeric test 2 (R406): the fp8 E4M3 grouped-verify path against a CPU reference.
//
// Links the standalone SM70_LONG_RAW object built from the vendored grouped-attention.cu
// and drives sm70_long_decode_fp8 on a small synthetic case, then compares every output
// element with a plain CPU softmax attention computed from the SAME dequantized KV.
//
// Every K/V value is exactly representable in E4M3 (+-0.5, +-1, +-2, +-4, 0) and every Q
// value is exactly representable in f16, so dequantization is lossless and the reference
// has no rounding ambiguity. Tolerance therefore only absorbs accumulation order.
//
// This single run settles: (a) the q / head-group index semantics, (b) the page table
// semantics, (c) the fp8 dequantization, (d) whether the path faults at all.
//
// Build:
//   nvcc -std=c++17 -arch=sm_70 -DSM70_LONG_RAW -I sm70-long -I . -O1 -c sm70-long/grouped-attention.cu -o /tmp/sm70fp8.o
//   nvcc -std=c++17 -arch=sm_70 -O2 fp8_verify_test.cu /tmp/sm70fp8.o -o /tmp/fp8test
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

extern "C" void sm70_long_decode_fp8(
    const void * q, const void * k_cache, const void * v_cache, void * out,
    const void * block_table, const void * row_lengths,
    void * partial, void * lse,
    int q_rows, int n_kv_heads, int n_q_heads_per_kv,
    int page_tokens, int n_pages,
    float k_scale, float v_scale, float softmax_scale, cudaStream_t stream);

static constexpr int D    = 256;
static constexpr int HQ   = 24;   // 4 KV heads x 6
static constexpr int HKV  = 4;
static constexpr int GQA  = HQ / HKV;
static constexpr int NQ   = 8;
static constexpr int KV   = 512;
static constexpr int PAGE = 256;
static constexpr int NP   = KV / PAGE;

// V must NOT be zero-mean: with a symmetric lattice the softmax-weighted average
// cancels towards zero, every relative error explodes at the cancellation, and the
// criterion becomes vacuous. This lattice is E4M3-exact and strongly positive.
static float kv_value(int t, int h, int d) {
    static const float lat[7] = { 0.25f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f, -1.0f };
    return lat[(t * 7 + h * 5 + d * 3) % 7];
}

int main() {
    const float scale = 1.0f / sqrtf((float) D);

    // ---- host inputs -----------------------------------------------------------------
    // The kernel's query addressing is compile time fixed and head major: plane j is
    // [h6][t][d], which is what the adapter's Q staging produces.
    std::vector<__half> hq((size_t) HKV * NQ * GQA * D);
    for (int j = 0; j < HKV; ++j)
        for (int t = 0; t < NQ; ++t)
            for (int h6 = 0; h6 < GQA; ++h6)
                for (int d = 0; d < D; ++d) {
                    static const float qlat[5] = { 0.25f, -0.25f, 0.5f, -0.5f, 0.125f };
                    const int h = j * GQA + h6;
                    hq[(((size_t) j * GQA + h6) * NQ + t) * D + d] = __float2half_rn(qlat[(t + h + d) % 5]);
                }

    // flat [d][t][h] caches, one byte per element (the llama.cpp layout)
    std::vector<uint8_t> hk((size_t) D * KV * HKV), hv((size_t) D * KV * HKV);
    for (int t = 0; t < KV; ++t)
        for (int h = 0; h < HKV; ++h)
            for (int d = 0; d < D; ++d) {
                const size_t idx = ((size_t) t * HKV + h) * D + d;
                hk[idx] = (uint8_t) __nv_cvt_float_to_fp8(kv_value(t, h, d),       __NV_SATFINITE, __NV_E4M3);
                hv[idx] = (uint8_t) __nv_cvt_float_to_fp8(kv_value(d, h, t + 11),  __NV_SATFINITE, __NV_E4M3);
            }

    std::vector<int32_t> bt(NP);
    for (int i = 0; i < NP; ++i) bt[i] = i;
    std::vector<int32_t> rl(1, KV);

    // ---- device buffers --------------------------------------------------------------
    void *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr;
    void *dbt = nullptr, *drl = nullptr, *dpart = nullptr, *dlse = nullptr;
    const size_t part_sz = (size_t) 80 * 8 * 6 * D;
    const size_t lse_sz  = (size_t) 80 * 8 * 6;
    cudaMalloc(&dq, hq.size() * sizeof(__half));
    cudaMalloc(&dk, hk.size());
    cudaMalloc(&dv, hv.size());
    cudaMalloc(&dout, hq.size() * sizeof(__half));
    cudaMalloc(&dbt, bt.size() * sizeof(int32_t));
    cudaMalloc(&drl, rl.size() * sizeof(int32_t));
    cudaMalloc(&dpart, part_sz * sizeof(float));
    cudaMalloc(&dlse,  lse_sz  * sizeof(float));
    cudaMemcpy(dq, hq.data(), hq.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk.data(), hk.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv.data(), hv.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dbt, bt.data(), bt.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(drl, rl.data(), rl.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

    sm70_long_decode_fp8(dq, dk, dv, dout, dbt, drl, dpart, dlse,
                         NQ, HKV, GQA, PAGE, NP, 1.0f, 1.0f, scale, 0);
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 2;
    }

    std::vector<__half> got((size_t) NQ * HQ * D);
    cudaMemcpy(got.data(), dout, got.size() * sizeof(__half), cudaMemcpyDeviceToHost);

    // ---- CPU reference ---------------------------------------------------------------
    int   worst_q = -1, worst_h = -1, worst_d = -1;
    double worst = 0.0, worst_rel = 0.0;
    // Near-zero outputs make a relative criterion meaningless, so grade twice: all
    // elements, and only those the attention actually populates (|ref| > 0.05).
    int    n_bad_all = 0, n_bad_sig = 0, n_sig = 0, n_bad_abs = 0;
    double worst_sig_rel = 0.0, worst_sig_abs = 0.0;
    int    ws_q = -1, ws_h = -1, ws_d = -1;
    for (int t = 0; t < NQ; ++t) {
        for (int h = 0; h < HQ; ++h) {
            const int j   = h / GQA;
            const int h6  = h % GQA;
            const int hkv = j;
            const size_t qo = (((size_t) j * GQA + h6) * NQ + t) * D;
            std::vector<float> score(KV);
            float mx = -1e30f;
            for (int s = 0; s < KV; ++s) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) {
                    dot += __half2float(hq[qo + d]) * kv_value(s, hkv, d);
                }
                score[s] = dot * scale;
                mx = std::max(mx, score[s]);
            }
            float sum = 0.0f;
            for (int s = 0; s < KV; ++s) { score[s] = expf(score[s] - mx); sum += score[s]; }
            for (int s = 0; s < KV; ++s) { score[s] /= sum; }
            for (int d = 0; d < D; ++d) {
                float ref = 0.0f;
                for (int s = 0; s < KV; ++s) { ref += score[s] * kv_value(d, hkv, s + 11); }
                const float g = __half2float(got[qo + d]);
                if (d == 0 && t < 2 && h < 6) {
                    printf("  [dump] j=%d h6=%d t=%d d=0  ref=%+.5f got=%+.5f\n", j, h6, t, ref, g);
                }
                const double e = fabs((double) g - (double) ref);
                const double r = e / (fabs((double) ref) + 1e-3);
                if (r > worst_rel) { worst_rel = r; worst = e; worst_q = t; worst_h = h; worst_d = d; }
                if (r > 2e-2) { ++n_bad_all; }
                if (e > 2e-2) { ++n_bad_abs; }
                if (fabs((double) ref) > 0.2) {
                    ++n_sig;
                    if (r > worst_sig_rel) { worst_sig_rel = r; worst_sig_abs = e; ws_q = t; ws_h = h; ws_d = d; }
                    if (r > 2e-2) { ++n_bad_sig; }
                }
            }
        }
    }

    // Grade on what the attention actually produces. A relative criterion is
    // meaningless for outputs near zero, where one f16 ULP dominates the ratio.
    const bool pass = (n_bad_abs == 0) && (n_bad_sig == 0);
    printf("FP8_VERIFY_TEST q=%d hq=%d hkv=%d kv=%d page=%d np=%d\n", NQ, HQ, HKV, KV, PAGE, NP);
    printf("  elements=%d significant_abs_gt_0.2=%d\n", NQ * HQ * D, n_sig);
    printf("  abs_err>2e-2: %d   sig_rel_err>2e-2: %d\n", n_bad_abs, n_bad_sig);
    printf("  worst_sig_rel=%.5g (abs=%.5g) at (q=%d, head=%d, d=%d)\n",
           worst_sig_rel, worst_sig_abs, ws_q, ws_h, ws_d);
    printf("  worst_all_rel=%.5g (abs=%.5g) at (q=%d, head=%d, d=%d)\n",
           worst_rel, worst, worst_q, worst_h, worst_d);
    printf("  => %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
