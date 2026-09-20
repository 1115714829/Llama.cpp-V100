// hmma_probe.cu -- determine the exact mma.m8n8k4.row.col.f32.f16.f16.f32 register layout
// empirically on this GPU.  No documentation trust: A and B fragments are filled per-thread
// with an unambiguous pattern, and every D register of every lane is dumped.
//
// Probe 1: a = {1,2,4,8}, b = {1,1,1,1}  -> D = sum_k a[k] = 15 for the row(s) this lane feeds
// Probe 2: a = {8,4,2,1}, b = {1,1,1,1}  -> same, reveals k order (1*8+2*4+4*2+8*1 = 8+8+8+8=32? no)
// Probe 3: a = {1,0,0,0}, b = {1,1,1,1}  -> only k=0 contributes
// Probe 4: a = {1,1,1,1}, b = {1,2,4,8}  -> reveals B's role
// Each probe writes 32 lanes x 8 D regs to a separate row of the output.
#include <cstdio>
#include <cuda_fp16.h>

#define WARP 32

__device__ __forceinline__ void mma884(float d[8], const unsigned a[2], const unsigned b[2], const float c[8]) {
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

__device__ __forceinline__ unsigned pack2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<unsigned *>(&h);
}

// probe id selects the A/B content; dumps 32x8 floats at out[probe*256]
__global__ void probe_kernel(float * out) {
    const int lane = threadIdx.x % WARP;
    float d[8], c[8];
    for (int i = 0; i < 8; ++i) { d[i] = 0.0f; c[i] = 0.0f; }

    for (int p = 0; p < 4; ++p) {
        unsigned a[2], b[2];
        if (p == 0) { a[0] = pack2(1, 2);      a[1] = pack2(4, 8); }
        if (p == 1) { a[0] = pack2(8, 4);      a[1] = pack2(2, 1); }
        if (p == 2) { a[0] = pack2(1, 0);      a[1] = pack2(0, 0); }
        if (p == 3) { a[0] = pack2(1, 1);      a[1] = pack2(1, 1); }
        if (p == 0 || p == 1 || p == 2) { b[0] = pack2(1, 1); b[1] = pack2(1, 1); }
        if (p == 3) { b[0] = pack2(1, 2);      b[1] = pack2(4, 8); }

        mma884(d, a, b, c);
        // dump
        for (int i = 0; i < 8; ++i) {
            out[(p * WARP + lane) * 8 + i] = d[i];
        }
        for (int i = 0; i < 8; ++i) { c[i] = 0.0f; d[i] = 0.0f; }
    }
}

int main() {
    float * d_out = nullptr;
    cudaMalloc(&d_out, 4 * WARP * 8 * sizeof(float));
    cudaMemset(d_out, 0, 4 * WARP * 8 * sizeof(float));
    probe_kernel<<<1, WARP>>>(d_out);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }

    float h[4 * WARP * 8];
    cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost);

    const char * names[4] = {
        "P0 a={1,2,4,8} b={1,1,1,1}",
        "P1 a={8,4,2,1} b={1,1,1,1}",
        "P2 a={1,0,0,0} b={1,1,1,1}",
        "P3 a={1,1,1,1} b={1,2,4,8}" };

    for (int p = 0; p < 4; ++p) {
        printf("\n=== %s ===\n", names[p]);
        printf("lane : d0 d1 d2 d3 d4 d5 d6 d7\n");
        for (int L = 0; L < WARP; ++L) {
            printf("%4d :", L);
            for (int i = 0; i < 8; ++i) {
                printf(" %6.1f", h[(p * WARP + L) * 8 + i]);
            }
            printf("\n");
        }
    }
    cudaFree(d_out);
    printf("\nPROBE_DONE\n");
    return 0;
}
