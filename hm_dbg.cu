// hm_dbg.cu -- instrumented epilogue: print (lane, reg, asup, bsup, d[r]) for every lane.
#include <cstdio>
#include <cuda_fp16.h>

#define WARP 32
#define QK8_0 32
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

__device__ __forceinline__ int asup(int L, int r) {
    const int base = 4 * (L >> 2) + (L & 1);
    return (r == 0 || r == 1 || r == 4 || r == 5) ? base : base + 2;
}
__device__ __forceinline__ int bsup(int L, int r) {
    const int c0 = 2 * ((L & 15) >> 1);
    if (r == 0 || r == 2) return c0;
    if (r == 1 || r == 3) return c0 + 1;
    if (r == 4 || r == 6) return c0 + 16;
    return c0 + 17;
}

// K = 32 (single Q8_0 block), A-lane L supplies token L%8, B-lane L supplies wrow L%8
__global__ void dbg_kernel(const __half * __restrict__ X, const block_q8_0_d * __restrict__ W,
                           float * __restrict__ out, int * __restrict__ info) {
    const int lane = threadIdx.x % WARP;
    const int token_a = lane % 8;
    const int wrow_b  = lane % 8;
    float d[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    const block_q8_0_d wb = W[wrow_b];
    const __half scale = wb.d;
    const __half * xb = X + (size_t)token_a * QK8_0;

    for (int k4 = 0; k4 < QK8_0; k4 += 4) {
        __half2 ah0 = __halves2half2(xb[k4 + 0], xb[k4 + 1]);
        __half2 ah1 = __halves2half2(xb[k4 + 2], xb[k4 + 3]);
        __half2 bh0 = __hmul2(__halves2half2(__int2half_rn(wb.qs[k4 + 0]), __int2half_rn(wb.qs[k4 + 1])), __half2half2(scale));
        __half2 bh1 = __hmul2(__halves2half2(__int2half_rn(wb.qs[k4 + 2]), __int2half_rn(wb.qs[k4 + 3])), __half2half2(scale));
        unsigned a[2] = { *reinterpret_cast<unsigned *>(&ah0), *reinterpret_cast<unsigned *>(&ah1) };
        unsigned b2[2] = { *reinterpret_cast<unsigned *>(&bh0), *reinterpret_cast<unsigned *>(&bh1) };
        mma884(d, a, b2);
    }

    for (int r = 0; r < 8; ++r) {
        info[lane * 16 + r * 2 + 0] = asup(lane, r);
        info[lane * 16 + r * 2 + 1] = bsup(lane, r);
        out[lane * 8 + r] = d[r];
    }
}

int main() {
    const int K = QK8_0;
    __half * hX = (__half *)malloc(sizeof(__half) * 8 * K);
    block_q8_0_d * hW = (block_q8_0_d *)malloc(sizeof(block_q8_0_d) * 8);
    srand(7);
    for (int i = 0; i < 8 * K; ++i) hX[i] = __float2half((float)((rand() % 21) - 10) * 0.25f);
    for (int n = 0; n < 8; ++n) {
        hW[n].d = __float2half(0.01f + (float)(rand() % 7) * 0.003f);
        for (int q = 0; q < K; ++q) hW[n].qs[q] = (signed char)((rand() % 33) - 16);
    }
    __half * dX; block_q8_0_d * dW; float * dOut; int * dInfo;
    cudaMalloc(&dX, sizeof(__half) * 8 * K);
    cudaMalloc(&dW, sizeof(block_q8_0_d) * 8);
    cudaMalloc(&dOut, sizeof(float) * WARP * 8);
    cudaMalloc(&dInfo, sizeof(int) * WARP * 16);
    cudaMemcpy(dX, hX, sizeof(__half) * 8 * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, hW, sizeof(block_q8_0_d) * 8, cudaMemcpyHostToDevice);
    dbg_kernel<<<1, WARP>>>(dX, dW, dOut, dInfo);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
    float hOut[WARP * 8]; int hInfo[WARP * 16];
    cudaMemcpy(hOut, dOut, sizeof(hOut), cudaMemcpyDeviceToHost);
    cudaMemcpy(hInfo, dInfo, sizeof(hInfo), cudaMemcpyDeviceToHost);

    printf("lane : r0..r7  (asup/bsup/value)  nonzero-regs\n");
    for (int L = 0; L < WARP; ++L) {
        printf("%4d :", L);
        for (int r = 0; r < 8; ++r) {
            printf(" %2d/%2d=%7.3f", hInfo[L * 16 + r * 2], hInfo[L * 16 + r * 2 + 1], hOut[L * 8 + r]);
        }
        printf("\n");
    }
    free(hX); free(hW);
    cudaFree(dX); cudaFree(dW); cudaFree(dOut); cudaFree(dInfo);
    printf("DBG_DONE\n");
    return 0;
}
