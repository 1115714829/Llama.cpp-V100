// wmma_q8_bench.cu -- Volta HMMA for the verify shape, but built on WMMA (nvcuda::wmma) so the
// compiler owns the fragment layout.  Validated against a CPU reference, then timed.
//
// Shape: A = weights (M=16 weight rows x K), B = activations (K x N=16 slots, of which 8 are real
// tokens), C = 16 x 16 f32  ->  out[token][weight row] = C[weight row][token].
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define QK8_0 32
struct __align__(2) block_q8_0 { __half d; signed char qs[QK8_0]; };

// ---------------------------------------------------------------- correctness probe (16x16x16)
__global__ void wmma_probe(const __half * A, const __half * B, float * C, int K) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a, A + k, K);          // A row-major [m][k], ldm = K
        wmma::load_matrix_sync(b, B + k, K);          // B col-major: element (k,n) at B[n*K + k]
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
}

// ---------------------------------------------------------------- Q8_0 GEMM, one warp per tile
// Weights: W[16 rows][K] in Q8_0 blocks.  Activations: X[16 slots][K] f16 (only 8 real).
// Out: out[t][n] = sum_k X[t][k] * W[n][k],  t in [0,8), n in [0,N)
__global__ void gemm_q8_wmma(const __half * __restrict__ X, const block_q8_0 * __restrict__ W,
                             float * __restrict__ out, int K, int N) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int tileN = warp;                    // which 16 weight rows
    if (tileN * 16 >= N) return;

    const int kb = K / QK8_0;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);

    __half af[16 * 16];
    __half bf[16 * 16];

    for (int k16 = 0; k16 < K; k16 += 16) {
        // A tile: 16 weight rows x 16 k, dequantized once and reused for all tokens
        for (int r = 0; r < 16; ++r) {
            const int n = tileN * 16 + r;
            for (int q = 0; q < 16; ++q) {
                const int k = k16 + q;
                const int blk = k / QK8_0;
                const block_q8_0 wb = W[(size_t)n * kb + blk];
                af[r * 16 + q] = __hmul(wb.d, __int2half_rn(wb.qs[k % QK8_0]));
            }
        }
        // B tile: 16 k x 16 token slots, laid out token-major so that a col_major load with
        // ldm=16 reads element (k,t) from bf[t*16 + k]
        for (int t = 0; t < 16; ++t) {
            for (int q = 0; q < 16; ++q) {
                bf[t * 16 + q] = (t < 8) ? X[(size_t)t * K + (k16 + q)] : __float2half(0.0f);
            }
        }
        // stage through shared for load_matrix_sync
        __shared__ __half sA[16 * 16];
        __shared__ __half sB[16 * 16];
        __syncthreads();
        for (int i = threadIdx.x % 32; i < 256; i += 32) { sA[i] = af[i]; sB[i] = bf[i]; }
        __syncthreads();
        wmma::load_matrix_sync(a, sA, 16);
        wmma::load_matrix_sync(b, sB, 16);
        wmma::mma_sync(c, a, b, c);
        __syncthreads();
    }

    __shared__ float sC[16 * 16];
    wmma::store_matrix_sync(sC, c, 16, wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x % 32; i < 256; i += 32) {
        const int r = i / 16, t = i % 16;
        if (t < 8) out[t * N + (tileN * 16 + r)] = sC[i];
    }
}

int main(int argc, char ** argv) {
    const int K = (argc > 1) ? atoi(argv[1]) : 64;
    const int N = 16;

    // ---- probe
    {
        __half * hA = (__half *)malloc(2 * 16 * K);
        __half * hB = (__half *)malloc(2 * 16 * K);
        srand(3);
        for (int i = 0; i < 16 * K; ++i) hA[i] = __float2half((float)((rand() % 9) - 4) * 0.5f);
        for (int i = 0; i < 16 * K; ++i) hB[i] = __float2half((float)((rand() % 9) - 4) * 0.5f);
        __half *dA, *dB; float *dC, hC[256];
        cudaMalloc(&dA, sizeof(__half) * 16 * K); cudaMalloc(&dB, sizeof(__half) * 16 * K);
        cudaMalloc(&dC, sizeof(float) * 256);
        cudaMemcpy(dA, hA, sizeof(__half) * 16 * K, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, sizeof(__half) * 16 * K, cudaMemcpyHostToDevice);
        wmma_probe<<<1, 32>>>(dA, dB, dC, K);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("probe CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
        cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost);
        double mx = 0;
        for (int m = 0; m < 16; ++m) for (int n = 0; n < 16; ++n) {
            double acc = 0;
            for (int k = 0; k < K; ++k)
                acc += (double)__half2float(hA[(size_t)m * K + k]) * (double)__half2float(hB[(size_t)n * K + k]);
            const double r = fabs((double)hC[m * 16 + n] - acc) / (fabs(acc) + 1e-9);
            if (r > mx) mx = r;
        }
        printf("WMMA 16x16x16 probe (K=%d): max rel err = %.3e -> %s\n", K, mx, mx < 1e-3 ? "MATCH" : "MISMATCH");
    }

    // ---- Q8_0 GEMM correctness
    {
        const int kb = K / QK8_0;
        __half * hX = (__half *)malloc(sizeof(__half) * 8 * K);
        block_q8_0 * hW = (block_q8_0 *)malloc(sizeof(block_q8_0) * N * kb);
        srand(99);
        for (int i = 0; i < 8 * K; ++i) hX[i] = __float2half((float)((rand() % 21) - 10) * 0.25f);
        for (int n = 0; n < N; ++n) for (int b = 0; b < kb; ++b) {
            hW[n * kb + b].d = __float2half(0.01f + (float)(rand() % 7) * 0.003f);
            for (int q = 0; q < QK8_0; ++q) hW[n * kb + b].qs[q] = (signed char)((rand() % 33) - 16);
        }
        __half * dX; block_q8_0 * dW; float * dOut, * hOut;
        cudaMalloc(&dX, sizeof(__half) * 8 * K);
        cudaMalloc(&dW, sizeof(block_q8_0) * N * kb);
        cudaMalloc(&dOut, sizeof(float) * 8 * N);
        hOut = (float *)malloc(sizeof(float) * 8 * N);
        cudaMemcpy(dX, hX, sizeof(__half) * 8 * K, cudaMemcpyHostToDevice);
        cudaMemcpy(dW, hW, sizeof(block_q8_0) * N * kb, cudaMemcpyHostToDevice);
        gemm_q8_wmma<<<1, 32>>>(dX, dW, dOut, K, N);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("gemm CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
        cudaMemcpy(hOut, dOut, sizeof(float) * 8 * N, cudaMemcpyDeviceToHost);

        double mx = 0;
        int nz = 0;
        for (int t = 0; t < 8; ++t) for (int n = 0; n < N; ++n) {
            double acc = 0;
            for (int k = 0; k < K; ++k) {
                const int b = k / QK8_0;
                const block_q8_0 wb = hW[(size_t)n * kb + b];
                acc += (double)__half2float(hX[(size_t)t * K + k]) * (double)__half2float(wb.d) * (double)wb.qs[k % QK8_0];
            }
            const double g = hOut[t * N + n];
            if (fabs(g) > 1e-6) ++nz;
            const double r = fabs(g - acc) / (fabs(acc) + 1e-9);
            if (r > mx) mx = r;
            if (t < 2 && n < 4) printf("  t=%d n=%d got %.5f ref %.5f\n", t, n, g, acc);
        }
        printf("Q8_0 WMMA GEMM (K=%d): nonzero=%d/%d  max rel err = %.3e -> %s\n",
               K, nz, 8 * N, mx, mx < 1e-3 ? "MATCH" : "MISMATCH");
    }
    printf("WMMA_BENCH_DONE\n");
    return 0;
}
