// Standalone lab for a Volta (sm_70) Q8_0 small-batch GEMM on tensor cores.
//
// Target shape: W[m][k] in block_q8_0 (34 B / 32 weights), Y[8 tokens][k] in block_q8_1,
// out[m][8] fp32 -- the DFlash2 verify batch (n=8) that the MMVQ path runs at ~526 GB/s.
//
// Why: ncu on the live kernel (m=4096 k=14336 n=8) shows 26.16M warp instructions, 7.2
// instructions per dp4a, 46% issue occupancy and LSU-queue stalls -- instruction bound, not
// bandwidth bound (DRAM traffic is identical at n=1 and n=8). The mma path decodes each weight
// value once for all 8 tokens instead of once per (row, token).
//
// Layout contract (ggml/src/ggml-cuda/mma.cuh, Volta tiles):
//   A tile<32,4,half2>            : get_i(l) = lane, get_j(l) = l
//   B tile<8,4,half2,I_MIRRORED>  : get_i(l) = (lane/16)*4 + (lane%4), get_j(l) = l
//   D tile<32,8,float>            : get_i(l) = (l&2) + (lane&~2), get_j(l) = (lane&2) + (l&5)
//   mma(d, {a[0],a[1]}, {b[0],b[1]}); mma(d, {a[2],a[3]}, {b[2],b[3]});
//
// Build: nvcc -arch=sm_70 -O3 -o w8volta-bench w8volta-bench.cu
// Run  : ./w8volta-bench [m] [k] [nsplit] [iters]

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define QK8_0 32

struct block_q8_0 {
    __half  d;
    int8_t  qs[QK8_0];
};
struct block_q8_1 {
    __half2 ds;   // .x = d (scale), .y = s (d * sum(q))
    int8_t  qs[QK8_0];
};

__device__ __forceinline__ void volta_mma(float (&d)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    int * Di = reinterpret_cast<int *>(d);
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+r"(Di[0]), "+r"(Di[1]), "+r"(Di[2]), "+r"(Di[3]), "+r"(Di[4]), "+r"(Di[5]), "+r"(Di[6]), "+r"(Di[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// Decode 4 int8 codes (one uint32) into two half2: (q0,q2) and (q1,q3). The 0x6400|u identity
// holds 1024+u exactly for u in [0,256), so subtracting 1152 recovers the signed code; one
// __hmul2 applies the group scale. Same routine for weights and activations, so the register
// slot order matches on both operands.
__device__ __forceinline__ void decode4(uint32_t raw, __half2 sc, __half2 bias, __half2 & lo, __half2 & hi) {
    const uint32_t u = raw ^ 0x80808080u;
    const uint32_t l = (u & 0x00FF00FFu) | 0x64006400u;
    const uint32_t h = ((u >> 8) & 0x00FF00FFu) | 0x64006400u;
    lo = __hmul2(__hsub2(*reinterpret_cast<const __half2 *>(&l), bias), sc);
    hi = __hmul2(__hsub2(*reinterpret_cast<const __half2 *>(&h), bias), sc);
}

// block_q8_0 is 34 bytes and qs starts at offset 2, so 4-byte accesses into it are misaligned
// for half of the blocks -- a plain uint32 load faults. Same workaround as llama.cpp get_int_b2.
__device__ __forceinline__ uint32_t load_b2(const int8_t * p, int i32) {
    const uint16_t * p16 = reinterpret_cast<const uint16_t *>(p);
    return (uint32_t) p16[2 * i32 + 0] | ((uint32_t) p16[2 * i32 + 1] << 16);
}

#define XS_PAD 8   // activation row: 32 halves + 8 pad = 40 halves = 80 B (16-byte aligned)
#define WS_PAD 8   // weight row:     32 halves + 8 pad = 40 halves = 80 B (16-byte aligned)

__device__ __forceinline__ void mma_pair(float (&d)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    volta_mma(d, a0, a1, b0, b1);
}

// Prefetch one k-block of raw codes: 8 weight words (4 codes each) + 8 scales for this lane's
// 4-row group, and 2 activation words + the token scale. Loads only, so the next iteration's
// memory latency overlaps this iteration's decode and mma.
struct raw_blk {
    uint32_t w[8];
    uint32_t y[2];
    __half   wsc[8];
    __half   ysc;
};

__device__ __forceinline__ void prefetch_blk(const block_q8_0 * __restrict__ W,
                                            const block_q8_1 * __restrict__ Y,
                                            int base, int m, int kblocks, int kb,
                                            int crow, int ccol, int ytok, int q4,
                                            raw_blk & r) {
    const uint32_t * yq = reinterpret_cast<const uint32_t *>(Y[(size_t) ytok * kblocks + kb].qs);
    r.ysc = __low2half(Y[(size_t) ytok * kblocks + kb].ds);
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        r.y[t] = yq[q4 + 4 * t];
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int row = base + ((j << 2) | crow);
        if (row < m) {
            const block_q8_0 * wb = &W[(size_t) row * kblocks + kb];
            r.w[j]   = load_b2(wb->qs, ccol >> 2);
            r.wsc[j] = wb->d;
        } else {
            r.w[j]   = 0x80808080u;   // decodes to 0
            r.wsc[j] = __ushort_as_half(0);
        }
    }
}

// One warp = 32 output rows. gridDim.y splits the K range; partials go to part[split][m][8].
//
// The weights are NOT loaded lane-per-row: 32 lanes reading 32 rows that are 15 KB apart costs
// 16.8 L1 sectors per warp load (measured: 33 M sectors for 63 MB of weights) and the kernel
// stalls on the LSU queue. Instead 8 lanes cooperate on one row (4 codes each) so a warp
// instruction touches 4 rows = 4-8 sectors, the decode lands in shared memory, and each lane then
// reads its own row back as four 16-byte chunks for the mma A fragments. The raw codes are
// prefetched one k-block ahead, otherwise the 128-warp grid is pure latency (13% issue).
__global__ __launch_bounds__(32, 16)
void w8volta_split(const block_q8_0 * __restrict__ W, const block_q8_1 * __restrict__ Y,
                   float * __restrict__ part, int m, int kblocks, int ntok, int kb_per_split) {
    const int lane = threadIdx.x & 31;
    const int base = blockIdx.x * 32;
    const int tok  = lane >> 2;      // 8 tokens x 4 lanes
    const int q4   = lane & 3;       // which uint32 of the 32 codes

    __shared__ __align__(16) __half xs[8][QK8_0 + XS_PAD];
    __shared__ __align__(16) __half ws[32][QK8_0 + WS_PAD];

    const __half2 bias = __half2half2(__ushort_as_half(0x6480));   // 1152.0
    float d[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    const int kb0 = blockIdx.y * kb_per_split;
    const int kb1 = min(kb0 + kb_per_split, kblocks);
    if (kb0 >= kb1) {
        return;
    }

    const int crow = lane >> 3;        // row inside the iteration's 4-row group
    const int ccol = (lane & 7) * 4;   // code offset inside the row

    raw_blk cur, nxt;
    prefetch_blk(W, Y, base, m, kblocks, kb0, crow, ccol, tok, q4, cur);

    for (int kb = kb0; kb < kb1; ++kb) {
        if (kb + 1 < kb1) {
            prefetch_blk(W, Y, base, m, kblocks, kb + 1, crow, ccol, tok, q4, nxt);
        }

        {
            const __half2 ysc = __half2half2(cur.ysc);
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int wi = q4 + 4 * t;
                __half2 lo, hi;
                decode4(cur.y[t], ysc, bias, lo, hi);
                *reinterpret_cast<__half2 *>(&xs[tok][wi * 4 + 0]) = lo;
                *reinterpret_cast<__half2 *>(&xs[tok][wi * 4 + 2]) = hi;
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int r = (j << 2) | crow;
                __half2 lo, hi;
                decode4(cur.w[j], __half2half2(cur.wsc[j]), bias, lo, hi);
                *reinterpret_cast<__half2 *>(&ws[r][ccol + 0]) = lo;
                *reinterpret_cast<__half2 *>(&ws[r][ccol + 2]) = hi;
            }
        }
        __syncwarp();

        const uint4 * Ap = reinterpret_cast<const uint4 *>(&ws[lane][0]);
        const uint4 A0 = Ap[0], A1 = Ap[1], A2 = Ap[2], A3 = Ap[3];
        const int btok = (lane >> 4) * 4 + (lane & 3);
        const uint4 * Bp = reinterpret_cast<const uint4 *>(&xs[btok][0]);
        const uint4 B0 = Bp[0], B1 = Bp[1], B2 = Bp[2], B3 = Bp[3];

        mma_pair(d, A0.x, A0.y, B0.x, B0.y);
        mma_pair(d, A0.z, A0.w, B0.z, B0.w);
        mma_pair(d, A1.x, A1.y, B1.x, B1.y);
        mma_pair(d, A1.z, A1.w, B1.z, B1.w);
        mma_pair(d, A2.x, A2.y, B2.x, B2.y);
        mma_pair(d, A2.z, A2.w, B2.z, B2.w);
        mma_pair(d, A3.x, A3.y, B3.x, B3.y);
        mma_pair(d, A3.z, A3.w, B3.z, B3.w);
        __syncwarp();

        cur = nxt;
    }

    float * dst = part + (size_t) blockIdx.y * m * ntok;
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int r = (l & 2) + (lane & ~2);
        const int c = (lane & 2) + (l & 5);
        if (base + r < m) {
            dst[(size_t) (base + r) * ntok + c] = d[l];
        }
    }
}

__global__ void w8volta_reduce(const float * __restrict__ part, float * __restrict__ out,
                               int n, int nsplit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float s = 0.0f;
    for (int k = 0; k < nsplit; ++k) {
        s += part[(size_t) k * n + i];
    }
    out[i] = s;
}

// reference: one thread per (row, token), fp64 accumulation over dequantized values
__global__ void ref_kernel(const block_q8_0 * __restrict__ W, const block_q8_1 * __restrict__ Y,
                           float * __restrict__ out, int m, int kblocks, int ntok) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= ntok) {
        return;
    }
    double acc = 0.0;
    for (int kb = 0; kb < kblocks; ++kb) {
        const block_q8_0 * wb = &W[(size_t) i * kblocks + kb];
        const block_q8_1 * yb = &Y[j * kblocks + kb];
        const double dw = (double) __half2float(wb->d);
        const double dy = (double) __half2float(__low2half(yb->ds));
        for (int t = 0; t < QK8_0; ++t) {
            acc += dw * (double) wb->qs[t] * dy * (double) yb->qs[t];
        }
    }
    out[(size_t) i * ntok + j] = (float) acc;
}

static void check(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        printf("CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

static float frand() {
    return (float) ((rand() / (double) RAND_MAX) * 2.0 - 1.0);
}

static void quantize_row_q8_1(const float * x, block_q8_1 * y, int k) {
    const int nb = k / QK8_0;
    for (int b = 0; b < nb; ++b) {
        float amax = 0.0f;
        for (int t = 0; t < QK8_0; ++t) {
            amax = fmaxf(amax, fabsf(x[b * QK8_0 + t]));
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        int sum = 0;
        for (int t = 0; t < QK8_0; ++t) {
            int q = (int) lroundf(x[b * QK8_0 + t] * id);
            q = q > 127 ? 127 : (q < -128 ? -128 : q);
            y[b].qs[t] = (int8_t) q;
            sum += q;
        }
        y[b].ds = __floats2half2_rn(d, d * (float) sum);
    }
}

int main(int argc, char ** argv) {
    const int m       = argc > 1 ? atoi(argv[1]) : 4096;
    const int k       = argc > 2 ? atoi(argv[2]) : 14336;
    const int nsplit  = argc > 3 ? atoi(argv[3]) : 8;
    const int iters   = argc > 4 ? atoi(argv[4]) : 200;
    const int ntok    = 8;
    const int kblocks = k / QK8_0;
    const int kb_per_split = (kblocks + nsplit - 1) / nsplit;

    printf("shape m=%d k=%d n=%d kblocks=%d nsplit=%d kb_per_split=%d\n",
           m, k, ntok, kblocks, nsplit, kb_per_split);

    std::vector<block_q8_0> hW((size_t) m * kblocks);
    std::vector<float>      hX((size_t) ntok * k);
    std::vector<block_q8_1> hY((size_t) ntok * kblocks);

    const bool probe = (getenv("PROBE") != nullptr);
    if (probe) {
        for (size_t i = 0; i < hW.size(); ++i) {
            hW[i].d = __float2half(1.0f);
            for (int t = 0; t < QK8_0; ++t) {
                hW[i].qs[t] = 0;
            }
        }
        for (int i = 0; i < m; ++i) {
            hW[(size_t) i * kblocks + 0].qs[i % QK8_0] = 1;
        }
        for (size_t i = 0; i < hY.size(); ++i) {
            hY[i].ds = __floats2half2_rn(1.0f, 0.0f);
            for (int t = 0; t < QK8_0; ++t) {
                hY[i].qs[t] = 0;
            }
        }
        for (int j = 0; j < ntok; ++j) {
            hY[(size_t) j * kblocks + 0].qs[4 * j] = 1;
        }
    } else {
        srand(1234);
        for (size_t i = 0; i < hW.size(); ++i) {
            const float d = 0.002f + 0.001f * (float) (rand() % 7);
            hW[i].d = __float2half(d);
            for (int t = 0; t < QK8_0; ++t) {
                hW[i].qs[t] = (int8_t) ((rand() % 255) - 127);
            }
        }
        for (int j = 0; j < ntok; ++j) {
            for (int t = 0; t < k; ++t) {
                hX[(size_t) j * k + t] = frand();
            }
            quantize_row_q8_1(&hX[(size_t) j * k], &hY[(size_t) j * kblocks], k);
        }
    }

    block_q8_0 * dW = nullptr;
    block_q8_1 * dY = nullptr;
    float * dpart = nullptr;
    float * dout = nullptr;
    float * dref = nullptr;
    check(cudaMalloc(&dW, hW.size() * sizeof(block_q8_0)), "malloc W");
    check(cudaMalloc(&dY, hY.size() * sizeof(block_q8_1)), "malloc Y");
    check(cudaMalloc(&dpart, (size_t) nsplit * m * ntok * sizeof(float)), "malloc part");
    check(cudaMalloc(&dout, (size_t) m * ntok * sizeof(float)), "malloc out");
    check(cudaMalloc(&dref, (size_t) m * ntok * sizeof(float)), "malloc ref");
    check(cudaMemcpy(dW, hW.data(), hW.size() * sizeof(block_q8_0), cudaMemcpyHostToDevice), "copy W");
    check(cudaMemcpy(dY, hY.data(), hY.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice), "copy Y");

    dim3 rblock(32, 8);
    dim3 rgrid((m + 31) / 32, (ntok + 7) / 8);
    ref_kernel<<<rgrid, rblock>>>(dW, dY, dref, m, kblocks, ntok);

    dim3 grid((m + 31) / 32, nsplit);
    w8volta_split<<<grid, 32>>>(dW, dY, dpart, m, kblocks, ntok, kb_per_split);
    const int nout = m * ntok;
    w8volta_reduce<<<(nout + 255) / 256, 256>>>(dpart, dout, nout, nsplit);
    check(cudaDeviceSynchronize(), "sync");

    std::vector<float> hout((size_t) m * ntok), href((size_t) m * ntok);
    check(cudaMemcpy(hout.data(), dout, hout.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out");
    check(cudaMemcpy(href.data(), dref, href.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy ref");

    double maxabs = 0.0, maxrel = 0.0, refscale = 0.0;
    for (size_t i = 0; i < hout.size(); ++i) {
        refscale = fmax(refscale, fabs((double) href[i]));
    }
    for (size_t i = 0; i < hout.size(); ++i) {
        const double e = fabs((double) hout[i] - (double) href[i]);
        maxabs = fmax(maxabs, e);
        if (fabs((double) href[i]) > 1e-3 * refscale) {
            maxrel = fmax(maxrel, e / fabs((double) href[i]));
        }
    }
    printf("correctness: maxabs=%.6g maxrel=%.3g (ref max |out|=%.6g)\n", maxabs, maxrel, refscale);

    if (probe) {
        for (int j = 0; j < ntok; ++j) {
            printf("probe j=%d (k=%2d): i =", j, 4 * j);
            for (int i = 0; i < m; ++i) {
                if (fabsf(hout[(size_t) i * ntok + j]) > 0.5f) {
                    printf(" %d", i);
                }
            }
            printf("   [ref:%d]\n", 4 * j);
        }
    }

    cudaEvent_t e0, e1;
    check(cudaEventCreate(&e0), "ev0");
    check(cudaEventCreate(&e1), "ev1");
    for (int i = 0; i < 10; ++i) {
        w8volta_split<<<grid, 32>>>(dW, dY, dpart, m, kblocks, ntok, kb_per_split);
        w8volta_reduce<<<(nout + 255) / 256, 256>>>(dpart, dout, nout, nsplit);
    }
    check(cudaEventRecord(e0), "rec0");
    for (int i = 0; i < iters; ++i) {
        w8volta_split<<<grid, 32>>>(dW, dY, dpart, m, kblocks, ntok, kb_per_split);
        w8volta_reduce<<<(nout + 255) / 256, 256>>>(dpart, dout, nout, nsplit);
    }
    check(cudaEventRecord(e1), "rec1");
    check(cudaEventSynchronize(e1), "sync1");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
    const double us = ms * 1000.0 / iters;
    const double bytes = (double) m * kblocks * sizeof(block_q8_0);
    printf("w8volta    : %8.2f us/run  %7.1f GB/s (weights only)  %6.1f GFLOP/s  grid=(%d,%d)\n",
           us, bytes / us / 1e3, 2.0 * m * k * ntok / us / 1e3, (m + 31) / 32, nsplit);

    check(cudaFree(dW), "free");
    check(cudaFree(dY), "free");
    check(cudaFree(dpart), "free");
    check(cudaFree(dout), "free");
    check(cudaFree(dref), "free");
    return 0;
}
