// hmma_map.cu -- extract the complete m8n8k4 index mapping by VALUE ENCODING.
//
// The D value is the product/sum of the A and B elements the hardware pairs up, so if I put a
// per-lane unique value into exactly one fragment and 1.0 everywhere in the other, then every
// nonzero D value *is* the id of the lane that supplied that row (or column).  Reading the whole
// 32x8 D register file therefore yields, in one run each:
//   run A: (lane -> matrix row m)  and the set of (lane,reg) slots owned by each row
//   run B: (lane -> matrix column n) and the slots owned by each column
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

// probe: kind 0 -> A carries lane ids, B = ones ; kind 1 -> A = ones, B carries lane ids
__global__ void map_kernel(float * out, int kind) {
    const int lane = threadIdx.x % WARP;
    const float id = (float)(lane + 1);
    __half2 z = __floats2half2_rn(0.0f, 0.0f);
    __half2 o = __floats2half2_rn(1.0f, 1.0f);

    unsigned a[2] = { *reinterpret_cast<unsigned *>(&o), *reinterpret_cast<unsigned *>(&o) };
    unsigned b[2] = { *reinterpret_cast<unsigned *>(&o), *reinterpret_cast<unsigned *>(&o) };

    __half2 idz = __floats2half2_rn(id, 0.0f);
    unsigned ida = *reinterpret_cast<unsigned *>(&idz);

    if (kind == 0) { a[0] = ida; a[1] = *reinterpret_cast<unsigned *>(&z); }  // A = {id, 0, 0, 0}
    else           { b[0] = ida; b[1] = *reinterpret_cast<unsigned *>(&z); }  // B = {id, 0, 0, 0}

    float d[8];
    mma884(d, a, b);
    for (int i = 0; i < 8; ++i) out[lane * 8 + i] = d[i];
}

int main() {
    float * dev = nullptr;
    cudaMalloc(&dev, WARP * 8 * sizeof(float));
    float h[WARP * 8];

    const char * nm[2] = { "A carries lane id (-> row map)", "B carries lane id (-> column map)" };
    for (int kind = 0; kind < 2; ++kind) {
        cudaMemset(dev, 0, WARP * 8 * sizeof(float));
        map_kernel<<<1, WARP>>>(dev, kind);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
        cudaMemcpy(h, dev, sizeof(h), cudaMemcpyDeviceToHost);

        printf("\n===== %s =====\n", nm[kind]);
        printf("lane :  d0   d1   d2   d3   d4   d5   d6   d7   (value = supplying lane + 1)\n");
        for (int L = 0; L < WARP; ++L) {
            printf("%4d :", L);
            for (int i = 0; i < 8; ++i) printf(" %4.0f", h[L * 8 + i]);
            printf("\n");
        }
    }
    cudaFree(dev);
    printf("\nMAP_DONE\n");
    return 0;
}
