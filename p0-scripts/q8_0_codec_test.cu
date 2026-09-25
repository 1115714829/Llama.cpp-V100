// q8_0 KV codec numeric test (R394).
//
// Links against the standalone SM70_LONG_RAW object built from the vendored
// grouped-attention.cu and runs its q8_0 probe kernel on a synthetic cache, then
// compares every element against a CPU reference. This pins the codec (block
// addressing + scale) and the flat-cache row contract (index = row*256 + d,
// row stride 272 bytes) before anything touches the serving machine.
//
// Build:
//   nvcc -std=c++17 -arch=sm_70 -DSM70_LONG_RAW -I sm70-long -I . -O1 -c sm70-long/grouped-attention.cu -o /tmp/sm70long.o
//   nvcc -std=c++17 -arch=sm_70 -O2 q8_0_codec_test.cu /tmp/sm70long.o -o /tmp/q8test
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>

extern "C" __global__ void sm70_long_q8_0_probe(const void * kv_cache, float * out, const int n);

static constexpr int kRowBytes = 272;  // 8 blocks of (half scale + 32 int8)

int main() {
    const int n_rows = 3;
    const int n_elem = n_rows * 256;
    std::vector<uint8_t> host(n_rows * kRowBytes);

    for (int r = 0; r < n_rows; ++r) {
        const float scale = 0.5f + 0.25f * (float) r;
        const __half hscale = __float2half_rn(scale);
        for (int b = 0; b < 8; ++b) {
            uint8_t * blk = host.data() + r * kRowBytes + b * 34;
            *reinterpret_cast<__half *>(blk) = hscale;
            int8_t * qs = reinterpret_cast<int8_t *>(blk + 2);
            for (int i = 0; i < 32; ++i) {
                const int d = b * 32 + i;
                qs[i] = (int8_t) (((r * 7 + d) % 255) - 127);
            }
        }
    }

    void * dev = nullptr;
    float * dev_out = nullptr;
    cudaMalloc(&dev, host.size());
    cudaMalloc(&dev_out, n_elem * sizeof(float));
    cudaMemcpy(dev, host.data(), host.size(), cudaMemcpyHostToDevice);

    sm70_long_q8_0_probe<<<(n_elem + 127) / 128, 128>>>((const void *) dev, dev_out, n_elem);
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 2;
    }

    std::vector<float> got(n_elem);
    cudaMemcpy(got.data(), dev_out, n_elem * sizeof(float), cudaMemcpyDeviceToHost);

    int bad = 0;
    double worst = 0.0;
    int worst_i = -1;
    for (int r = 0; r < n_rows; ++r) {
        const float scale = 0.5f + 0.25f * (float) r;
        for (int d = 0; d < 256; ++d) {
            const int i = r * 256 + d;
            const int q = ((r * 7 + d) % 255) - 127;
            // the loader converts through half, so compare against the half-rounded product
            const float ref = __half2float(__float2half_rn(scale * (float) q));
            const float diff = fabsf(got[i] - ref);
            if (diff > worst) { worst = diff; worst_i = i; }
            if (diff > 1e-3f) { ++bad; }
        }
    }

    printf("Q8_0_CODEC_TEST rows=%d elems=%d mismatches=%d max_abs_diff=%.6g (at %d) => %s\n",
           n_rows, n_elem, bad, worst, worst_i, bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
