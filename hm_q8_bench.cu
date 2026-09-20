// hm_q8_bench.cu -- Volta SM70 (HMMA m8n8k4) Q8_0 GEMM for the speculative-verify shape.
//
// Goal: measure what the M<=8 quantized GEMM can reach when the weights are dequantized once
// and reused across all 8 tokens via the FP16 tensor core, instead of the current
// per-column dp4a path.
//
// The fragment index mapping below is NOT from documentation: it was measured on this GPU with
// one-hot / value-encoded probes (see hmma_layout.cu, hmma_map.cu).  Values reported by those
// probes are lane+1, and the composed result is:
//   regs {d0,d1,d4,d5} of lane L hold row  aSup = 4*(L>>2) + (L&1)
//   regs {d2,d3,d6,d7} of lane L hold row  aSup + 2
//   columns: c0 = 2*((L&15)>>1);  {d0,d2}<->c0, {d1,d3}<->c0+1, {d4,d6}<->c0+16, {d5,d7}<->c0+17
// with A-lane L supplying token (L%8) and B-lane L supplying weight row (L%8).
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define WARP 32
#define QK8_0 32

// Q8_0 block: fp16 scale + 32 int8
struct __align__(2) block_q8_0_d { __half d; signed char qs[QK8_0]; };

__device__ __forceinline__ void mma884(float d[8], const unsigned a[2], const unsigned b[2]) {
    float c[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"
        "{%0,%1,%2,%3,%4,%5,%6,%7},"
        "{%8,%9},"
        "{%10,%11},"
        "{%12,%13,%14,%15,%16,%17,%18,%19};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]),
          "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
}

// amap / bmap as measured (lane that supplies the row / column of each register)
__device__ __host__ __forceinline__ int asup(int L, int r) {
    const int base = 4 * (L >> 2) + (L & 1);
    return (r == 0 || r == 1 || r == 4 || r == 5) ? base : base + 2;
}
__device__ __host__ __forceinline__ int bsup(int L, int r) {
    const int c0 = 2 * ((L & 15) >> 1);
    if (r == 0 || r == 2) return c0;
    if (r == 1 || r == 3) return c0 + 1;
    if (r == 4 || r == 6) return c0 + 16;
    return c0 + 17;
}

// One warp computes out[8 tokens][8 weight rows] for k = 0..K-1.
// X: activations in f16, laid out [token][k]      (row-major, stride K)
// W: weights Q8_0,        laid out [wrow][k]      (row-major, k blocks of 32)
// out: [token][wrow] f32, stride 8
__global__ void gemm_hmma_kernel(const __half * __restrict__ X, const block_q8_0_d * __restrict__ W,
                                 float * __restrict__ out, int K) {
    const int lane = threadIdx.x % WARP;
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const int wrow_base = wid * 8;          // this warp's 8 weight rows
    const int kb = K / QK8_0;               // number of Q8_0 blocks per row

    const int token_a = lane % 8;           // A-lane supplies this token
    const int wrow_b  = lane % 8;           // B-lane supplies this weight row

    float d[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int b = 0; b < kb; ++b) {
        const block_q8_0_d wb = W[(size_t)(wrow_base + wrow_b) * kb + b];
        const __half scale = wb.d;
        const __half * xb = X + (size_t)token_a * K + b * QK8_0;

        for (int k4 = 0; k4 < QK8_0; k4 += 4) {
            // A fragment: 4 halves = activations k4..k4+3 of this lane's token
            __half2 ah0 = __halves2half2(xb[k4 + 0], xb[k4 + 1]);
            __half2 ah1 = __halves2half2(xb[k4 + 2], xb[k4 + 3]);
            // B fragment: 4 halves = dequantized weights, dequantized once and reused by all 8 tokens
            __half2 bh0 = __hmul2(__halves2half2(__int2half_rn(wb.qs[k4 + 0]), __int2half_rn(wb.qs[k4 + 1])), __half2half2(scale));
            __half2 bh1 = __hmul2(__halves2half2(__int2half_rn(wb.qs[k4 + 2]), __int2half_rn(wb.qs[k4 + 3])), __half2half2(scale));

            unsigned a[2] = { *reinterpret_cast<unsigned *>(&ah0), *reinterpret_cast<unsigned *>(&ah1) };
            unsigned bfr[2] = { *reinterpret_cast<unsigned *>(&bh0), *reinterpret_cast<unsigned *>(&bh1) };
            mma884(d, a, bfr);
        }
    }

    // epilogue: only one of the 4 replications writes
    for (int r = 0; r < 8; ++r) {
        const int s = asup(lane, r);
        if (s >= 8) continue;                       // deduplicate the 4 hardware replications
        const int token = s % 8;
        const int wrow  = bsup(lane, r) % 8;
        out[token * 8 + wrow] = d[r];
    }
}

// CPU reference for the same tile
static void ref_gemm(const float * X, const block_q8_0_d * W, float * out, int K) {
    for (int t = 0; t < 8; ++t) {
        for (int n = 0; n < 8; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                const int b = k / QK8_0;
                const block_q8_0_d wb = W[(size_t)n * (K / QK8_0) + b];
                const float s = __half2float(wb.d) * (float)wb.qs[k % QK8_0];
                acc += (double)__half2float(X[(size_t)t * K + k]) * (double)s;
            }
            out[t * 8 + n] = (float)acc;
        }
    }
}

int main(int argc, char ** argv) {
    const int K = (argc > 1) ? atoi(argv[1]) : 64;     // correctness check with small K
    const int N = 8;                                    // one tile = 8 weight rows
    const int kb = K / QK8_0;

    __half * hX = (__half *)malloc(sizeof(__half) * 8 * K);
    block_q8_0_d * hW = (block_q8_0_d *)malloc(sizeof(block_q8_0_d) * N * kb);
    srand(1234);
    for (int i = 0; i < 8 * K; ++i) hX[i] = __float2half((float)((rand() % 21) - 10) * 0.25f);
    for (int n = 0; n < N; ++n) for (int b = 0; b < kb; ++b) {
        hW[n * kb + b].d = __float2half(0.01f + (float)(rand() % 7) * 0.003f);
        for (int q = 0; q < QK8_0; ++q) hW[n * kb + b].qs[q] = (signed char)((rand() % 33) - 16);
    }

    __half * dX; block_q8_0_d * dW; float * dOut, * hOut, * hRef;
    cudaMalloc(&dX, sizeof(__half) * 8 * K);
    cudaMalloc(&dW, sizeof(block_q8_0_d) * N * kb);
    cudaMalloc(&dOut, sizeof(float) * 64);
    hOut = (float *)malloc(sizeof(float) * 64);
    hRef = (float *)malloc(sizeof(float) * 64);
    cudaMemcpy(dX, hX, sizeof(__half) * 8 * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, hW, sizeof(block_q8_0_d) * N * kb, cudaMemcpyHostToDevice);

    gemm_hmma_kernel<<<1, WARP>>>(dX, dW, dOut, K);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(hOut, dOut, sizeof(float) * 64, cudaMemcpyDeviceToHost);

    // reference from the same half inputs, accumulated in double
    float * Xf = (float *)malloc(sizeof(float) * 8 * K);
    for (int i = 0; i < 8 * K; ++i) Xf[i] = __half2float(hX[i]);
    block_q8_0_d * Wtmp = hW;
    for (int t = 0; t < 8; ++t) for (int n = 0; n < 8; ++n) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
            const int b = k / QK8_0;
            const block_q8_0_d wb = Wtmp[(size_t)n * kb + b];
            const double s = (double)__half2float(wb.d) * (double)wb.qs[k % QK8_0];
            acc += (double)Xf[(size_t)t * K + k] * s;
        }
        hRef[t * 8 + n] = (float)acc;
    }

    printf("=== correctness (K=%d)  out[t][n] ===\n", K);
    double maxrel = 0.0;
    for (int t = 0; t < 8; ++t) {
        for (int n = 0; n < 8; ++n) {
            const float g = hOut[t * 8 + n], r = hRef[t * 8 + n];
            const double rel = (r != 0.0f) ? fabs((double)g - (double)r) / fabs((double)r) : fabs((double)g);
            if (rel > maxrel) maxrel = rel;
            printf("%9.4f/%9.4f ", g, r);
        }
        printf("\n");
    }
    printf("max relative error = %.3e  -> %s\n", maxrel, (maxrel < 1e-3) ? "MATCH" : "MISMATCH");

    free(hX); free(hW); free(hOut); free(hRef); free(Xf);
    cudaFree(dX); cudaFree(dW); cudaFree(dOut);
    printf("BENCH_CHECK_DONE\n");
    return 0;
}
