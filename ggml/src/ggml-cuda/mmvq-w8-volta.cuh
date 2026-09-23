#pragma once

// Q8_0 x Q8_1 mat-vec on Volta tensor cores (mma.sync.m8n8k4), sm_70 only.
// Both operands decode into fp16 through the 0x6400 identity, the k loop runs inside the
// global latency behind a double buffer, and each warp owns 8 output rows outright.

#include "common.cuh"
#include "ggml.h"

#include <cuda_fp16.h>
#include <cstdint>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

namespace {

struct w8v_sched {
    // one warp per CTA: small decode shapes (m=4096) need the CTA count for latency hiding
    static constexpr int kWarps     = 1;
    static constexpr int kKStep     = 32;   // == QK8_0: one block per staging step
    static constexpr int kTTile     = 32;   // rows of the mma A operand (m8n8k4 quadpairs)
    static constexpr int kXPad      = 8;
    static constexpr int kRowsPerCta = kWarps * 8;
    static constexpr int kThreads   = kWarps * 32;
};

// m8n8k4 fragment addressing (warp-wide: 4 quadpairs, each an 8x8 tile at k4)
__device__ __forceinline__ int w8v_qp_row()  { return threadIdx.x & 31; }
__device__ __forceinline__ int w8v_k_row() {
    const int lane = threadIdx.x & 31;
    return ((lane / 16) * 4) + (lane % 4);
}
__device__ __forceinline__ int w8v_d_i(int l) {
    const int lane = threadIdx.x & 31;
    return (l & 2) + (lane & ~2);
}
__device__ __forceinline__ int w8v_d_j(int l) {
    const int lane = threadIdx.x & 31;
    return (lane & 2) + (l & (4 + 1));
}

__device__ __forceinline__ void w8v_load_qp(half2 (&dst)[4], const half2 * base, int stride) {
    const int row = w8v_qp_row();
    *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(base + row * stride);
}

__device__ __forceinline__ void w8v_load_k(half2 (&dst)[4], const half2 * base, int stride) {
    const int row = w8v_k_row();
    *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(base + row * stride);
}

// D[32x8 f32] += A[32x8 half] @ B[8x8 half]^T, real k = 8 per call (two m8n8k4 at k4)
__device__ __forceinline__ void w8v_mma(float (&d)[8], const half2 (&a)[4], const half2 (&b)[4]) {
    const int * Axi = reinterpret_cast<const int *>(a);
    const int * Bxi = reinterpret_cast<const int *>(b);
    int * Dxi       = reinterpret_cast<int *>(d);
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3]), "+r"(Dxi[4]),
                   "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
                 : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]), "r"(Bxi[1]));
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3]), "+r"(Dxi[4]),
                   "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
                 : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[2]), "r"(Bxi[3]));
}

// eight int8 codes in p8 -> four half2 of (code * sc). the 0x6400|u trick: u = b^0x80 in
// [0,255] makes 0x6400|u equal 1152+b as fp16, so one hsub2 vs 1152.0 recovers the code.
__device__ __forceinline__ void w8v_decode8(uint2 p8, float sc_f, half2 (&out)[4]) {
    const half2 sc2  = __half2half2(__float2half(sc_f));
    const half2 bias = __half2half2(__ushort_as_half(0x6480)); // 1152.0
    const uint32_t w0 = p8.x ^ 0x80808080u;
    const uint32_t w1 = p8.y ^ 0x80808080u;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t src = (j < 2) ? w0 : w1;
        const int shift    = (j & 1) * 16;
        uint32_t bits = (((src >> shift) & 0xffu) | (((src >> shift) & 0xff00u) << 8)) | 0x64006400u;
        out[j] = __hmul2(__hsub2(*reinterpret_cast<half2 *>(&bits), bias), sc2);
    }
}

struct w8v_carry {
    uint2 xp8;   // 8 activation codes for this lane's step
    uint2 wp8;   // 8 weight codes
    float xsc;
    float wsc;
    bool  xactive;
};

} // namespace

// x: q8_1 columns (t columns of k elems, col stride x_stride bytes).
// w: q8_0 rows (n rows of k elems, row stride w_row_blocks blocks).
// out: [n][t] fp32 with row stride out_ld (== number of columns).
static __global__ __launch_bounds__(w8v_sched::kThreads, 15) void w8v_mmvq_q8_0_kernel(
    const void * __restrict__ w_ptr, const void * __restrict__ x_ptr, float * __restrict__ out_ptr,
    int n, int k, int t, int64_t w_row_blocks, int64_t x_stride, int64_t out_ld) {
    using S = w8v_sched;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int n0   = (static_cast<int>(blockIdx.x) * S::kWarps + warp) * 8;
    const int t0   = static_cast<int>(blockIdx.y) * S::kTTile;
    const int tcnt = min(S::kTTile, t - t0);

    // x_sh is per warp (tiny): no cross-warp hazards, so the pipeline needs no block barrier
    __shared__ __align__(16) __half x_sh[2][S::kWarps][S::kTTile][S::kKStep + S::kXPad];
    __shared__ __align__(16) __half w_sh[2][S::kWarps][8][S::kKStep + S::kXPad];

    constexpr int kVecs = S::kKStep / 8;

    auto prefetch = [&](int kbase, w8v_carry & r) {
        const int idx  = static_cast<int>(threadIdx.x) & 31;
        r.xactive      = idx < tcnt * kVecs;
        r.xp8          = make_uint2(0, 0);
        r.wp8          = make_uint2(0, 0);
        r.xsc          = 0.0f;
        r.wsc          = 0.0f;
        if (r.xactive) {
            const int row = idx / kVecs;
            const int v   = idx % kVecs;
            const int g   = kbase / S::kKStep;
            const block_q8_1 * col = reinterpret_cast<const block_q8_1 *>(
                static_cast<const char *>(x_ptr) + static_cast<int64_t>(t0 + row) * x_stride);
            const block_q8_1 * b = col + g;
            const uint32_t * p = reinterpret_cast<const uint32_t *>(b->qs + v * 8);
            r.xp8 = make_uint2(p[0], p[1]);
            r.xsc = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t *>(b)));
        }
        const int g2  = kbase / S::kKStep;
        const int row = n0 + (lane >> 2);
        if (row < n) {
            const block_q8_0 * wb = static_cast<const block_q8_0 *>(w_ptr) +
                static_cast<int64_t>(row) * w_row_blocks + g2;
            // only u16 is safe here: q8_0 blocks are 34 B so base+2 shifts alignment per block
            const uint16_t * q16 = reinterpret_cast<const uint16_t *>(wb->qs + (lane & 3) * 8);
            r.wp8 = make_uint2(q16[0] | ((uint32_t) q16[1] << 16), q16[2] | ((uint32_t) q16[3] << 16));
            r.wsc = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t *>(wb)));
        }
    };

    auto commit = [&](const w8v_carry & r, int buf) {
        if (r.xactive) {
            const int idx = static_cast<int>(threadIdx.x) & 31;
            __align__(16) half2 dec[4];
            w8v_decode8(r.xp8, r.xsc, dec);
            *reinterpret_cast<uint4 *>(&x_sh[buf][warp][idx / kVecs][(idx % kVecs) * 8]) =
                *reinterpret_cast<const uint4 *>(dec);
        }
        const int r_row = lane >> 2;
        const int b0    = (lane & 3) * 8;
        __align__(16) half2 wdec[4];
        w8v_decode8(r.wp8, r.wsc, wdec);
        *reinterpret_cast<uint4 *>(&w_sh[buf][warp][r_row][b0]) =
            *reinterpret_cast<const uint4 *>(wdec);
    };

    float d[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    w8v_carry carry;
    prefetch(0, carry);
    commit(carry, 0);
    int buf = 0;

    // one barrier per iteration: i reads buf and writes buf^1, so the top barrier separates
    // iteration i-1's reads from iteration i's writes
    for (int kbase = 0; kbase < k; kbase += S::kKStep) {
        __syncwarp();
        const int nxt       = kbase + S::kKStep;
        const bool has_next = nxt < k;
        if (has_next) { prefetch(nxt, carry); }

#pragma unroll
        for (int kk = 0; kk < S::kKStep; kk += 8) {
            __align__(16) half2 a[4];
            __align__(16) half2 b[4];
            w8v_load_qp(a, reinterpret_cast<const half2 *>(&x_sh[buf][warp][0][kk]),
                        (S::kKStep + S::kXPad) / 2);
            w8v_load_k(b, reinterpret_cast<const half2 *>(&w_sh[buf][warp][0][kk]),
                       (S::kKStep + S::kXPad) / 2);
            w8v_mma(d, a, b);
        }

        if (has_next) { commit(carry, buf ^ 1); }
        buf ^= 1;
    }

#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int row_t = w8v_d_i(l);
        const int col_n = w8v_d_j(l);
        const int nn    = n0 + col_n;
        const int tt    = t0 + row_t;
        if (row_t < tcnt && nn < n) {
            out_ptr[static_cast<int64_t>(tt) * out_ld + nn] = d[l];
        }
    }
}

#endif // sm_70
