#include <cuda_runtime.h>
#include <cstdlib>

// R350: 2D device-to-device interleave for the DFlash feature injection.
// The injection batch is [token][layer][dim] while the extract tensors are
// [layer][token][dim]; one 2D copy per layer writes the interleaved rows.
extern "C" int ggml_cuda_copy2d(void * dst, size_t dst_pitch,
                                const void * src, size_t src_pitch,
                                size_t width_bytes, size_t rows) {
    const cudaError_t err = cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch,
                                              width_bytes, rows,
                                              cudaMemcpyDeviceToDevice, 0);
    return (int) err;
}

extern "C" void * ggml_cuda_alloc_bytes(size_t bytes, int device) {
    if (device >= 0) {
        cudaSetDevice(device);
    }
    void * p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        return nullptr;
    }
    return p;
}

extern "C" void ggml_cuda_free_bytes(void * p, int device) {
    if (p == nullptr) {
        return;
    }
    if (device >= 0) {
        cudaSetDevice(device);
    }
    cudaFree(p);
}

extern "C" int ggml_cuda_stream_sync() {
    return (int) cudaStreamSynchronize(0);
}
