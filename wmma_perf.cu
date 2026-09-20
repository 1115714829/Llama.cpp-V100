// wmma_perf.cu -- measure what the WMMA (HMMA) Q8_0 verify-shape GEMM actually achieves,
// so the M<=8 ceiling can be compared against the current dp4a path's 340 GB/s.
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define QK8_0 32
struct __align__(2) block_q8_0 { __half d; signed char qs[QK8_0]; };

// One warp = 16 weight rows x 16 token slots (8 real).  K walked in 16-steps.
__global__ void gemm_q8_wmma(const __half * __restrict__ X, const block_q8_0 * __restrict__ W,
                             float * __restrict__ out, int K, int N) {
    const int warp  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane  = threadIdx.x % 32;
    const int tileN = warp;
    if (tileN * 16 >= N) return;
    const int kb = K / QK8_0;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);

    __shared__ __half sA[16 * 16];
    __shared__ __half sB[16 * 16];

    for (int k16 = 0; k16 < K; k16 += 16) {
        const int blk = k16 >> 5;          // QK8_0 = 32, and 16 divides 32
        const int q0  = k16 & 31;

        for (int i = lane; i < 256; i += 32) {
            const int r = i >> 4, q = i & 15;
            const block_q8_0 wb = W[(size_t)(tileN * 16 + r) * kb + blk];
            const signed char sv = (q0 + q < 32) ? wb.qs[q0 + q] : W[(size_t)(tileN*16+r)*kb + blk + 1].qs[(q0 + q) - 32];
            const __half sc = (q0 + q < 32) ? wb.d : W[(size_t)(tileN*16+r)*kb + blk + 1].d;
            sA[i] = __hmul(sc, __int2half_rn(sv));
        }
        for (int i = lane; i < 256; i += 32) {
            const int q = i >> 4, t = i & 15;
            sB[i] = (t < 8) ? X[(size_t)t * K + (k16 + q)] : __float2half(0.0f);
        }
        __syncwarp();
        wmma::load_matrix_sync(a, sA, 16);
        wmma::load_matrix_sync(b, sB, 16);
        wmma::mma_sync(c, a, b, c);
        __syncwarp();
    }

    __shared__ float sC[16 * 16];
    wmma::store_matrix_sync(sC, c, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < 256; i += 32) {
        const int r = i >> 4, t = i & 15;
        if (t < 8) out[(size_t)t * N + (tileN * 16 + r)] = sC[i];
    }
}

int main(int argc, char ** argv) {
    const int K = (argc > 1) ? atoi(argv[1]) : 5120;
    const int N = (argc > 2) ? atoi(argv[2]) : 17408;
    const int iters = (argc > 3) ? atoi(argv[3]) : 20;
    const int kb = K / QK8_0;

    __half * dX; block_q8_0 * dW; float * dOut;
    cudaMalloc(&dX, sizeof(__half) * 16 * K);
    cudaMalloc(&dW, (size_t)sizeof(block_q8_0) * N * kb);
    cudaMalloc(&dOut, sizeof(float) * 8 * N);
    cudaMemset(dW, 0x21, (size_t)sizeof(block_q8_0) * N * kb);   // any data; we only time it
    cudaMemset(dX, 0x11, sizeof(__half) * 16 * K);

    const int warps = (N + 15) / 16;
    const int threads = 256;
    const int blocks = (warps * 32 + threads - 1) / threads;

    for (int i = 0; i < 3; ++i) gemm_q8_wmma<<<blocks, threads>>>(dX, dW, dOut, K, N);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i) gemm_q8_wmma<<<blocks, threads>>>(dX, dW, dOut, K, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);

    const double wbytes = (double)N * K * (double)sizeof(block_q8_0) / QK8_0;   // Q8_0 payload
    const double per_us = ms * 1000.0 / iters;
    const double gbps = wbytes / (per_us * 1e-6) / 1e9;
    const double macs = 8.0 * N * K;
    const double tflops = 2.0 * macs / (per_us * 1e-6) / 1e12;

    printf("=== WMMA Q8_0 verify-shape GEMM ===\n");
    printf("N=%d K=%d tokens=8  warps=%d blocks=%d\n", N, K, warps, blocks);
    printf("weight bytes = %.1f MB\n", wbytes / 1e6);
    printf("%.1f us/run   ->   %.1f GB/s effective   (%.1f TFLOPS)\n", per_us, gbps, tflops);
    printf("reference: current dp4a path measured 340 GB/s at n=8 (n=1 roofline 729 GB/s)\n");
    printf("WMMA_PERF_DONE\n");
    return 0;
}
