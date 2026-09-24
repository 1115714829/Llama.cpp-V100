// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// fattn-sm70-decomp.cuh - P-P3 decomposition compute (T2 core).
//
// Dataflow (p3-decomp spec [S2], mirrors 1cat 79T):
//   per KV-head group (gqa=6 Q heads packed into M, zero-copy: the 6 heads are
//   contiguous in GGML Q, so the group IS a [256, q6] column-major matrix):
//     QK  : S[m, j] = Q[d, m] . K[d, j]     one cuBLAS GEMM per kBlockN block
//     soft: exact fp32 block softmax merge (running row max/sum; P -> f16)
//     PV  : O[d, m] += V[d, j] * P[m, j]    one cuBLAS GEMM per block
//     epi : O *= 1/sum -> f32 output (GGML [256(d), q_len, hq] layout)
//
// Layout convention (single, checked): everywhere column-major cuBLAS style.
//   S/P      [q6, kbn]  : elem (m, j) at j*q6 + m
//   O / out  [256, q6]  : elem (d, m) at d + m*256
//   mask     [n_kv, q_len] (GGML ne0 = n_kv contiguous) : (j, t) at j + t*n_kv
// Score/row-state live in the private resident workspace (never in the graph).
// Numerics: fp32 score + exact max subtraction (no f16 score range risk).
// ============================================================================
#pragma once

#include "fattn-sm70-d256-kernel.cuh"

#include <cublas_v2.h>

namespace FLASH_NAMESPACE {

using index_t = uint32_t;

// mask is F16 when flash_attn is on (build_attn_inp_kq_mask), F32 otherwise.
__device__ __forceinline__ float sm70_decomp_mask_at(
        const void * mask, bool mask_f16, index_t idx) {
    return mask_f16 ? __half2float(((const half *) mask)[idx])
                    : ((const float *) mask)[idx];
}

// S/P are [q6, kbn] (elem (m,j) at j*q6+m); O is [256, q6] (d + m*256).
// One block of kbn KV rows per launch; running row max/sum merges blocks.
template <int kQ6PerBlk>
__global__ void sm70_decomp_softmax_block_kernel(
        const float * __restrict__ S,       // [q6, kbn] raw dots (unscaled)
        half        * __restrict__ P,       // [q6, kbn]
        float       * __restrict__ row_max, // [q6] raw-dot domain
        float       * __restrict__ row_sum, // [q6]
        float       * __restrict__ O,       // [256, q6]
        const void  * __restrict__ mask,    // [n_kv, q_len], (j,t) at j + t*n_kv
        bool mask_f16,
        int kbn, int q6, int q_len, int n_kv, int j0, int gqa,
        float softmax_scale_log2, bool first_block) {
    const int m0 = (int) blockIdx.x * kQ6PerBlk;
    for (int mm = m0; mm < m0 + kQ6PerBlk && mm < q6; ++mm) {
        // packed column order is token-major: m = t*gqa + h (Q layout [d,h,q])
        const int t = mm / gqa;
        // 1) block max over raw dots + add-mask
        float bmax = -INFINITY;
        for (int j = 0; j < kbn; ++j) {
            const float mv =
                sm70_decomp_mask_at(mask, mask_f16, (j0 + j) + (index_t) t * n_kv);
            const float s = S[(index_t) j * q6 + mm] + mv;
            if (s > bmax) { bmax = s; }
        }
        float rmax = first_block ? -INFINITY : row_max[mm];
        float rsum = first_block ? 0.0f : row_sum[mm];
        if (bmax == -INFINITY) {
            // fully masked block (bucket padding tail): contributes nothing
            row_max[mm] = rmax;
            row_sum[mm] = rsum;
            continue;
        }
        // 2) running max update + O/sum rescale (factor <= 1)
        const float new_max = bmax > rmax ? bmax : rmax;
        if (new_max != rmax && rmax != -INFINITY) {
            const float factor = exp2f((rmax - new_max) * softmax_scale_log2);
            rsum *= factor;
            for (int d = 0; d < 256; ++d) {
                O[d + (index_t) mm * 256] *= factor;
            }
        }
        // 3) exp -> P (f16) relative to new_max + block sum
        float bsum = 0.0f;
        for (int j = 0; j < kbn; ++j) {
            const float mv =
                sm70_decomp_mask_at(mask, mask_f16, (j0 + j) + (index_t) t * n_kv);
            const float e = exp2f(
                (S[(index_t) j * q6 + mm] + mv - new_max) * softmax_scale_log2);
            P[(index_t) j * q6 + mm] = (half) e;
            bsum += e;
        }
        row_max[mm] = new_max;
        row_sum[mm] = rsum + bsum;
    }
}

// Pack kbn KV rows of one head into [256, kbn] col-major (f16). Source rows may
// interleave heads (measured: K nb=(34,544,272) => row r = 2*j + h), so gather
// by strides instead of assuming a packed head slice.
__global__ void sm70_decomp_pack_kv_kernel(
        const half * __restrict__ src,
        int src_rstep, int src_hstep, int h0,
        int j0, int kbn,
        half * __restrict__ dst) {
    for (int jj = (int) blockIdx.x * blockDim.x + threadIdx.x;
            jj < kbn; jj += (int) blockDim.x * gridDim.x) {
        const half * s = src
            + (index_t) (j0 + jj) * src_rstep * 256
            + (index_t) h0 * src_hstep * 256;
        half * d = dst + (index_t) jj * 256;
        for (int x = 0; x < 256; ++x) { d[x] = s[x]; }
    }
}

// Pack one KV-head group of Q into [256, q6] col-major (f16: cublasGemmEx
// requires uniform operand types). Source layout is GGML [d, h, q]:
// (d, t, h) at d + h*256 + t*hq*256 (measured, R299 diag).
// Column order c = t*gqa + hh (token-major) - matches O/mask conventions.
__global__ void sm70_decomp_pack_q_kernel(
        const void * __restrict__ Q,
        bool q_f16,
        int g0, int gqa, int q_len, int hq,
        int q_t, int q_h,
        half * __restrict__ Qp) {
    for (int c = (int) blockIdx.x * blockDim.x + threadIdx.x;
            c < q_len * gqa; c += (int) blockDim.x * gridDim.x) {
        const int t  = c / gqa;
        const int hh = c % gqa;
        const index_t src =
            (index_t) (g0 + hh) * q_h + (index_t) t * q_t;
        half * dst = Qp + (index_t) c * 256;
        if (q_f16) {
            const half * s = (const half *) Q + src;
            for (int d = 0; d < 256; ++d) { dst[d] = s[d]; }
        } else {
            const float * s = (const float *) Q + src;
            for (int d = 0; d < 256; ++d) { dst[d] = (half) s[d]; }
        }
    }
}

// O *= 1/row_sum -> scatter into dst using its REAL nb strides (dst may be
// canonical [d, q, h] or the [d, h, q] family - strides decide, never assume).
__global__ void sm70_decomp_epilogue_kernel(
        const float * __restrict__ O,
        float       * __restrict__ out,
        const float * __restrict__ row_sum,
        int q6, int gqa, int g0,
        int o_t, int o_h) {
    for (int mm = (int) blockIdx.x * blockDim.x + threadIdx.x;
            mm < q6; mm += (int) blockDim.x * gridDim.x) {
        const int t  = mm / gqa;
        const int hh = mm % gqa;
        const float inv = 1.0f / row_sum[mm];
        float * base = out + (index_t) (g0 + hh) * o_h + (index_t) t * o_t;
        for (int d = 0; d < 256; ++d) {
            base[d] = O[d + (index_t) mm * 256] * inv;
        }
    }
}

// Private cublas handles + resident workspaces, PER DEVICE (TP runs several
// devices concurrently; a single global pointer/handle cross-wires them).
constexpr int kDecompMaxDev = 16;

struct sm70_decomp_dev {
    cublasHandle_t cublas;
    void * ws;
    size_t ws_bytes;
};

inline sm70_decomp_dev & sm70_decomp_state(int id) {
    static sm70_decomp_dev devs[kDecompMaxDev] = {};
    return devs[id];
}

// Full decomposed attention for one KV-head group (6 packed Q heads).
// Qg/Kg/Vg are column-major [256, N] views (d contiguous); out_g is the
// matching [256, q6] f32 output slice. ws is the private resident workspace.
static void sm70_decomp_group(
        cudaStream_t stream,
        int dev_id,
        const void   * Q_raw,   // [256, q_len, hq] GGML layout (d,h,q)
        cudaDataType_t q_type,
        const half   * Kg,      // base of the whole K staging (strides below)
        const half   * Vg,
        float        * out_raw,
        float        * ws,
        const void   * mask,
        bool mask_f16,
        int q6, int q_len, int hq, int g0, int gkv, int kbn_total, int kBlockN, int gqa,
        int q_t, int q_h, int o_t, int o_h,
        int k_rstep, int k_hstep, int v_rstep, int v_hstep,
        float softmax_scale_log2) {
    const int n_kv = kbn_total;
    const int kbn_blk = kBlockN < kbn_total ? kBlockN : kbn_total;
    (void) kbn_blk;

    // workspace partition: Qp [256, q6] f16, Kp/Vp [256, kbn_blk] f16,
    // S/P [q6, kbn_blk], O [256, q6], row_max/row_sum [q6]
    half  * Qp      = (half *) ws;
    half  * Kp      = Qp + (size_t) 256 * q6;
    half  * Vp      = Kp + (size_t) 256 * kbn_blk;
    float * S       = (float *) (Vp + (size_t) 256 * kbn_blk);
    half  * P       = (half *) (S + (size_t) kbn_blk * q6);
    float * O       = (float *) (P + (size_t) kbn_blk * q6);
    float * row_max = O + (size_t) 256 * q6;
    float * row_sum = row_max + q6;

    sm70_decomp_pack_q_kernel<<<(unsigned) ((q6 + 255) / 256), 256, 0, stream>>>(
        Q_raw, q_type == CUDA_R_16F, g0, gqa, q_len, hq, q_t, q_h, Qp);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemsetAsync(row_sum, 0, (size_t) q6 * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(O, 0, (size_t) 256 * q6 * sizeof(float), stream));

    cublasHandle_t cublas = sm70_decomp_state(dev_id).cublas;
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    // R304 ceiling probe (GGML_DECOMP_TIMES=1): phase wall times around the
    // cuBLAS GEMMs at the production shape (the fused-form hardcap anchor).
    // ev1/ev2 get a fallback record before the loop: a zero-block call would
    // otherwise time unrecorded events (cudaErrorInvalidValue).
    const bool dprof = getenv("GGML_DECOMP_TIMES") != nullptr;
    cudaEvent_t ev0, ev1, ev2, ev3;
    if (dprof) {
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        cudaEventCreate(&ev2);
        cudaEventCreate(&ev3);
        cudaEventRecord(ev0, stream);
        cudaEventRecord(ev1, stream);
        cudaEventRecord(ev2, stream);
    }

    const float alpha = 1.0f;
    const float beta0 = 0.0f;
    const float beta1 = 1.0f;

    for (int j0 = 0; j0 < kbn_total; j0 += kBlockN) {
        const int kbn = min(kBlockN, kbn_total - j0);
        sm70_decomp_pack_kv_kernel<<<(unsigned) ((kbn + 255) / 256), 256, 0, stream>>>(
            Kg, k_rstep, k_hstep, gkv, j0, kbn, Kp);
        sm70_decomp_pack_kv_kernel<<<(unsigned) ((kbn + 255) / 256), 256, 0, stream>>>(
            Vg, v_rstep, v_hstep, gkv, j0, kbn, Vp);
        CUDA_CHECK(cudaGetLastError());
        if (dprof) {
            cudaEventRecord(ev1, stream);
        }

        // QK: S[q6, kbn] = Qp^T[q6, 256] x Kp[256, kbn], fp32 accumulate.
        CUBLAS_CHECK(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                q6, kbn, 256, &alpha,
                Qp, CUDA_R_16F, 256,
                Kp, CUDA_R_16F, 256,
                &beta0, S, CUDA_R_32F, q6,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        if (dprof) {
            cudaEventRecord(ev2, stream);
        }

        // block softmax + running merge (scale folded via scale_log2 here).
        sm70_decomp_softmax_block_kernel<32><<<
            (unsigned) ((q6 + 31) / 32), 128, 0, stream>>>(
                S, P, row_max, row_sum, O, mask, mask_f16,
                kbn, q6, q_len, n_kv, j0, gqa,
                softmax_scale_log2, j0 == 0);
        CUDA_CHECK(cudaGetLastError());
        if (dprof) {
            cudaEventRecord(ev3, stream);
        }

        // PV: O[256, q6] += Vp[256, kbn] x P^T[kbn, q6], fp32 accumulate.
        CUBLAS_CHECK(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                256, q6, kbn, &alpha,
                Vp, CUDA_R_16F, 256,
                P, CUDA_R_16F, q6,
                &beta1, O, CUDA_R_32F, 256,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        if (dprof) {
            cudaEventRecord(ev2, stream);
        }
    }

    sm70_decomp_epilogue_kernel<<<(unsigned) min(q6, 256), 32, 0, stream>>>(
        O, out_raw, row_sum, q6, gqa, g0, o_t, o_h);
    CUDA_CHECK(cudaGetLastError());
    if (dprof) {
        cudaEventRecord(ev3, stream);
        cudaEventSynchronize(ev3);
        float tpack, tqk, tsoft, tpv;
        cudaEventElapsedTime(&tpack, ev0, ev1);
        cudaEventElapsedTime(&tqk, ev1, ev2);
        cudaEventElapsedTime(&tsoft, ev2, ev3);
        cudaEventElapsedTime(&tpv, ev1, ev3);
        static int nd = 0;
        if (nd < 64 || (kbn_total > 200000 && nd < 96)) {
            nd++;
            fprintf(stderr, "[DPROF] pack=%.2f qk=%.2f soft=%.2f rest=%.2f tot=%.2f ms (q6=%d kbn=%d)\n",
                tpack, tqk, tsoft, tpv - tqk - tsoft, tpack + tpv, q6, kbn_total);
        }
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        cudaEventDestroy(ev2);
        cudaEventDestroy(ev3);
    }
}

} // namespace FLASH_NAMESPACE
