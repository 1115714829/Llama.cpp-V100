// hmma_layout.cu -- determine the exact mma.m8n8k4.row.col.f32.f16.f16.f32 fragment layout
// empirically.  Method: one-hot probing.
//   * A-probe : set exactly ONE half of ONE lane's A fragment to 1.0, B = all ones.
//               D[m][n] = 1 for every n of the single row m that this A slot feeds.
//   * B-probe : A = all ones, exactly ONE half of ONE lane's B fragment to 1.0.
//               D[m][n] = 1 for every m of the single column n that this B slot feeds.
// Each launch dumps all 32 lanes x 8 D registers as a 256-bit mask, so the mapping is read
// off directly instead of being inferred from documentation.
#include <cstdio>
#include <cuda_fp16.h>

#define WARP 32

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

__device__ __forceinline__ unsigned ones2() {
    __half2 h = __floats2half2_rn(1.0f, 1.0f);
    return *reinterpret_cast<unsigned *>(&h);
}

// one-hot: exactly the `slot`-th half of thread `sel`'s fragment is 1.0
__device__ __forceinline__ void set_one(unsigned f[2], int lane, int sel, int slot) {
    f[0] = 0u; f[1] = 0u;
    if (lane != sel) return;
    __half2 h = (slot < 2) ? __floats2half2_rn(slot == 0 ? 1.0f : 0.0f, slot == 1 ? 1.0f : 0.0f)
                           : __floats2half2_rn(slot == 2 ? 1.0f : 0.0f, slot == 3 ? 1.0f : 0.0f);
    f[slot / 2] = *reinterpret_cast<unsigned *>(&h);
}

__global__ void probeA(float * out, int sel, int slot) {
    const int lane = threadIdx.x % WARP;
    unsigned a[2], b[2] = { ones2(), ones2() };
    set_one(a, lane, sel, slot);
    float d[8];
    mma884(d, a, b);
    for (int i = 0; i < 8; ++i) out[lane * 8 + i] = d[i];
}

__global__ void probeB(float * out, int sel, int slot) {
    const int lane = threadIdx.x % WARP;
    unsigned a[2] = { ones2(), ones2() }, b[2];
    set_one(b, lane, sel, slot);
    float d[8];
    mma884(d, a, b);
    for (int i = 0; i < 8; ++i) out[lane * 8 + i] = d[i];
}

int main() {
    float * dev = nullptr;
    cudaMalloc(&dev, WARP * 8 * sizeof(float));
    float h[WARP * 8];

    const char * which[2] = { "A", "B" };
    for (int kind = 0; kind < 2; ++kind) {
        printf("\n########## %s-probe (one-hot in the %s fragment, other = all ones) ##########\n",
               which[kind], which[kind]);
        printf("format: sel.slot -> list of (lane,reg) that became 1.0\n");
        for (int sel = 0; sel < WARP; ++sel) {
            for (int slot = 0; slot < 4; ++slot) {
                cudaMemset(dev, 0, WARP * 8 * sizeof(float));
                if (kind == 0) probeA<<<1, WARP>>>(dev, sel, slot);
                else           probeB<<<1, WARP>>>(dev, sel, slot);
                cudaError_t e = cudaDeviceSynchronize();
                if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
                cudaMemcpy(h, dev, sizeof(h), cudaMemcpyDeviceToHost);

                printf("%2d.%d ->", sel, slot);
                int n = 0;
                for (int L = 0; L < WARP; ++L) {
                    for (int i = 0; i < 8; ++i) {
                        if (h[L * 8 + i] != 0.0f) { printf(" %d:%d=%.0f", L, i, h[L * 8 + i]); ++n; }
                    }
                }
                printf("   [%d hits]\n", n);
            }
        }
    }
    cudaFree(dev);
    printf("\nLAYOUT_DONE\n");
    return 0;
}
