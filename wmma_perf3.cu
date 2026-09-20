// wmma_perf3.cu -- double-buffered HMMA Q8_0 verify-shape GEMM.
// v2 was a pure latency chain: each k-step did a ~4.3 KB uncoalesced load plus syncs with no
// overlap.  v3 software-pipelines the staging (prefetch step k+1 while computing step k) and
// distributes the staging over all threads.
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define QK8_0 32
struct __align__(2) block_q8_0 { __half d; signed char qs[QK8_0]; };

#define ROWS 128                 // weight rows per block (8 warps x 16)
#define KC   64                  // k values staged per step (2 Q8_0 blocks)
#define NT   16                  // token slots (8 real)
#define NBLK (KC / QK8_0)        // Q8_0 blocks per staged step

__global__ void __launch_bounds__(256) gemm_q8_hmma3(
        const __half * __restrict__ X, const block_q8_0 * __restrict__ W,
        float * __restrict__ out, int K, int N) {
    __shared__ __half sW[2][ROWS][KC];
    __shared__ __half sX[2][NT][KC];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row0 = blockIdx.x * ROWS;
    const int kb   = K / QK8_0;
    const int nstep = kb / NBLK;

    // stage() fills buffer b with k-step s: one global Q8_0 block read per (row, blk)
    auto stage = [&](int b, int s) {
        const int kbase = s * KC;
        // weights: 128 rows x 2 blocks = 256 units, one per thread
        {
            const int r = tid >> 1;              // 0..127
            const int bi = tid & 1;              // 0..1  (which Q8_0 block)
            const int n = row0 + r;
            const int qoff = bi * QK8_0;
            if (n < N) {
                const block_q8_0 wb = W[(size_t)n * kb + (kbase / QK8_0) + bi];
                #pragma unroll
                for (int q = 0; q < QK8_0; ++q) sW[b][r][qoff + q] = __hmul(wb.d, __int2half_rn(wb.qs[q]));
            } else {
                #pragma unroll
                for (int q = 0; q < QK8_0; ++q) sW[b][r][qoff + q] = __float2half(0.0f);
            }
        }
        // activations: 16 slots x 64 k = 1024 elements, 4 per thread
        for (int i = tid; i < NT * KC; i += blockDim.x) {
            const int t = i / KC, q = i % KC;
            sX[b][t][q] = (t < 8) ? X[(size_t)t * K + kbase + q] : __float2half(0.0f);
        }
    };

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bfr;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    stage(0, 0);
    __syncthreads();

    for (int s = 0; s < nstep; ++s) {
        const int cur = s & 1, nxt = cur ^ 1;
        if (s + 1 < nstep) stage(nxt, s + 1);      // prefetch next step (no sync yet)

        #pragma unroll
        for (int half = 0; half < KC / 16; ++half) {
            wmma::load_matrix_sync(a,  &sW[cur][warp * 16][half * 16], KC);
            wmma::load_matrix_sync(bfr, &sX[cur][0][half * 16],        KC);
            wmma::mma_sync(acc, a, bfr, acc);
        }
        __syncthreads();                            // everyone done reading cur before restaging
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
    for (int i = 0; i < 3; ++i) gemm_q8_hmma3<<<blocks, 256>>>(dX, dW, dOut, K, N);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i) gemm_q8_hmma3<<<blocks, 256>>>(dX, dW, dOut, K, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);

    const double wbytes = (double)N * K * (double)sizeof(block_q8_0) / QK8_0;
    const double per_us = ms * 1000.0 / iters;
    printf("=== pipelined HMMA Q8_0 verify GEMM (v3) ===\n");
    printf("N=%d K=%d tokens=8 blocks=%d\n", N, K, blocks);
    printf("weight bytes = %.1f MB\n", wbytes / 1e6);
    printf("%.1f us/run  ->  %.1f GB/s effective\n", per_us, wbytes / (per_us * 1e-6) / 1e9);
    printf("targets: dp4a n=8 = 340 GB/s ; n=1 roofline = 729 GB/s\n");
    printf("WMMA_PERF3_DONE\n");
    return 0;
}
