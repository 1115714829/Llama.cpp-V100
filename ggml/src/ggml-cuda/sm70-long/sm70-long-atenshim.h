#pragma once

// Minimal ATen stand-in so the vendored 1cat grouped-attention launchers compile
// inside ggml-cuda without torch. Only the surface the launchers touch is provided:
// Tensor (size/sizes/stride/dim/data_ptr/scalar_type/device/is_contiguous), the
// scalar-type tags, IntArrayRef, TORCH_CHECK, the CUDA guard and stream/property
// accessors.
//
// This header is only used when SM70_LONG_RAW is defined; the upstream file keeps
// its original includes otherwise.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <initializer_list>
#include <vector>
#include <string>

namespace at {

using Half = __half;

enum ScalarType { kHalf, kByte, kInt, kFloat, kDouble, kBool, kLong };

class IntArrayRef {
public:
    IntArrayRef() = default;
    IntArrayRef(std::initializer_list<int64_t> v) : v_(v) {}
    IntArrayRef(const std::vector<int64_t> & v) : v_(v) {}
    int64_t size() const { return (int64_t) v_.size(); }
    int64_t operator[](size_t i) const { return v_[i]; }
    bool operator==(const IntArrayRef & o) const { return v_ == o.v_; }
    bool operator!=(const IntArrayRef & o) const { return !(*this == o); }
    size_t dim() const { return v_.size(); }
private:
    std::vector<int64_t> v_;
};

class Device {
public:
    Device() = default;
    explicit Device(int index) : index_(index) {}
    int index() const { return index_; }
    bool operator==(const Device & o) const { return index_ == o.index_; }
private:
    int index_ = 0;
};

class Tensor {
public:
    Tensor() = default;
    Tensor(void * data, ScalarType type, std::vector<int64_t> sizes, std::vector<int64_t> strides, int device)
        : data_(data), type_(type), sizes_(std::move(sizes)), strides_(std::move(strides)), device_(device) {}

    int dim() const { return (int) sizes_.size(); }
    int64_t size(int i) const { return sizes_[i]; }
    IntArrayRef sizes() const { return IntArrayRef(sizes_); }
    int64_t stride(int i) const { return strides_[i]; }
    ScalarType scalar_type() const { return type_; }
    bool is_contiguous() const { return true; }
    bool is_cuda() const { return true; }
    Device device() const { return Device(device_); }

    void * data_ptr() const { return data_; }

    template <typename T>
    T * data_ptr() const { return reinterpret_cast<T *>(data_); }

    // only the unused ATen entry touches clone(); return an alias so it compiles
    Tensor clone() const { return *this; }

    // the launchers only pass these through to the kernels
    Tensor & operator=(const Tensor &) = default;

private:
    void * data_ = nullptr;
    ScalarType type_ = kFloat;
    std::vector<int64_t> sizes_;
    std::vector<int64_t> strides_;
    int device_ = 0;
};

namespace cuda {

struct Stream {
    cudaStream_t s = nullptr;
    cudaStream_t stream() const { return s; }
};

inline Stream getCurrentCUDAStream() { return Stream{}; }

struct DeviceProperties {
    int major = 7;
    int minor = 0;
    int sharedMemPerBlockOptin = 96 * 1024;
};

inline const DeviceProperties * getCurrentDeviceProperties() {
    static DeviceProperties props;
    static bool inited = false;
    if (!inited) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, 0) == cudaSuccess) {
            props.major = p.major;
            props.minor = p.minor;
            props.sharedMemPerBlockOptin = (int) p.sharedMemPerBlockOptin;
        }
        inited = true;
    }
    return &props;
}

} // namespace cuda

} // namespace at

namespace c10 {
namespace cuda {

class CUDAGuard {
public:
    explicit CUDAGuard(const at::Device & dev) : prev_(0) { cudaGetDevice(&prev_); cudaSetDevice(dev.index()); }
    ~CUDAGuard() { cudaSetDevice(prev_); }
private:
    int prev_;
};

} // namespace cuda
} // namespace c10

#ifndef TORCH_CHECK
#define TORCH_CHECK(cond, ...)                                                  \
    do {                                                                        \
        if (!(cond)) {                                                          \
            fprintf(stderr, "[sm70-long] TORCH_CHECK failed: %s\n", #cond);     \
            abort();                                                            \
        }                                                                       \
    } while (0)
#endif

#ifndef TORCH_WARN
#define TORCH_WARN(...)                                                         \
    do {                                                                        \
        fprintf(stderr, "[sm70-long] WARN: ");                                  \
        fprintf(stderr, __VA_ARGS__);                                           \
        fprintf(stderr, "\n");                                                  \
    } while (0)
#endif

#ifndef C10_CUDA_CHECK
#define C10_CUDA_CHECK(expr)                                                    \
    do {                                                                        \
        const cudaError_t err_ = (expr);                                        \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "[sm70-long] CUDA error %d: %s at %s:%d\n",         \
                    (int) err_, cudaGetErrorString(err_), __FILE__, __LINE__);   \
            abort();                                                            \
        }                                                                       \
    } while (0)
#endif

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
#endif
