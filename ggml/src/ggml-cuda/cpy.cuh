#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

// R383: fused CPY + ADD (same-shape elementwise), one pass.
void ggml_cuda_op_cpy_add_fused(ggml_backend_cuda_context & ctx, ggml_tensor * cpy, ggml_tensor * add);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
