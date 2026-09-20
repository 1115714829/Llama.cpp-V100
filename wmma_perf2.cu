// wmma_perf2.cu -- proper smem-staged HMMA Q8_0 verify-shape GEMM.
// Each Q8_0 block is loaded from global exactly once, dequantized into shared, and reused by
// all 8 tokens.  This is the design that should approach the weight-read roofline.
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define QK8_0 32
struct __align__(2) block_q8_0 { __half d; signed char qs[QK8_0]; };

#define ROWS 128            // weight rows per block = 8 warps x 16
#define NT   16             // token slots (8 real)

__global__ void __launch_bounds__(256) gemm_q8_hmma(
        const __half * __restrict__ X, const block_q8_0 * __restrict__ W,
        float * __restrict__ out, int K, int N) {
    __shared__ __half    sW[ROWS][QK8_0];     // dequantized weights
    __shared__ __half    sX[NT][QK8_0];       // activations

    const int tid   = threadIdx.x;
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int row0  = blockIdx.x * ROWS;
    const int kb    = K / QK8_0;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int kc = 0; kc < kb; ++kc) {
        // stage weights: one global block read per row, then 32 dequantized halves in smem
        for (int i = tid; i < ROWS; i += blockDim.x) {
            const int n = row0 + i;
            if (n < N) {
                const block_q8_0 wb = W[(size_t)n * kb + kc];
                #pragma unroll
                for (int q = 0; q < QK8_0; ++q) sW[i][q] = __hmul(wb.d, __int2half_rn(wb.qs[q]));
            } else {
                #pragma unroll
                for (int q = 0; q < QK8_0; ++q) sW[i][q] = __float2half(0.0f);
            }
        }
        // stage activations (16 slots x 32 k); only 8 are real
        for (int i = tid; i < NT * QK8_0; i += blockDim.x) {
            const int t = i >> 5, q = i & 31;
            sX[t][q] = (t < 8) ? X[(size_t)t * K + kc * QK8_0 + q] : __float2half(0.0f);
        }
        __syncthreads();

        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            wmma::load_matrix_sync(a, &sW[warp * 16][half * 16], QK8_0);   // row-major, ldm = 32
            wmma::load_matrix_sync(b, &sX[0][half * 16], QK8_0);           // col-major, ldm = 32
            wmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }

    __shared__ float sC[ROWS][NT];
    wmma::store_matrix_sync(&sC[warp * 16][0], acc, NT, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < 16 * NT; i += 32) {
        const int r = i / NT, t = i % NT;
        if (t < 8) out[(size_t)t * N + (row0 + warp * 16 + r)] = sC[warp * 16 + r][t];
    }
}

int main(int argc, char ** argv) {
    const int K     = (argc > 1) ? atoi(argv[1]) : 5120;
    const int N     = (argc > 2) ? atoi(argv[2]) : 17408;
    const int iters = (argc > 3) ? atoi(argv[3]) : 20;
    const int kb    = K / QK8_0;

    __half * dX; block_q8_0 * dW; float * dOut;
    cudaMalloc(&dX, sizeof(__half) * NT * K);
    cudaMalloc(&dW, (size_t)sizeof(block_q8_0) * N * kb);
    cudaMalloc(&dOut, sizeof(float) * 8 * N);
    cudaMemset(dW, 0x21, (size_t)sizeof(block_q8_0) * N * kb);
    cudaMemset(dX, 0x11, sizeof(__half) * NT * K);

    const int blocks = (N + ROWS - 1) / ROWS;
    for (int i = 0; i < 3; ++i) gemm_q8_hmma<<<blocks, 256>>>(dX, dW, dOut, K, N);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i) gemm_q8_hmma<<<blocks, 256>>>(dX, dW, dOut, K, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);

    const double wbytes = (double)N * K * (double)sizeof(block_q8_0) / QK8_0;
    const double per_us = ms * 1000.0 / iters;
    printf("=== smem-staged HMMA Q8_0 verify GEMM ===\n");
    printf("N=%d K=%d tokens=8  blocks=%d\n", N, K, blocks);
    printf("weight bytes = %.1f MB\n", wbytes / 1e6);
    printf("%.1f us/run  ->  %.1f GB/s effective  (%.2f TFLOPS on 8 tokens)\n",
           per_us, wbytes / (per_us * 1e-6) / 1e9, 2.0 * 8.0 * N * K / (per_us * 1e-6) / 1e12);
    printf("targets: current dp4a n=8 = 340 GB/s ; n=1 roofline = 729 GB/s\n");
    printf("WMMA_PERF2_DONE\n");
    return 0;
}
