#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MMVQ wide vec_dot: VDR 2 -> 8 for Q8_0, so one call covers a whole 32-value block.

R259 (ncu) measured 7.2 thread-instructions per dp4a at n=8: the kernel spends ~5 of them per
call on the fp32 scale conversion, and a call only covers 8 of the 32 values of a block because
four threads share each block (qi/vdr = 4). With vdr = 8 a thread owns the whole block: 8 dp4a
per call, one scale multiply, and 8 uint32 activation loads instead of 4 x 2.

Env gated: GGML_CUDA_MMVQ_WIDE=1 -> wide path, unset -> today's code byte for byte.
Only instantiated for GGML_TYPE_Q8_0 / ncols_dst == 8.

Usage: wip-mmvq-wide.py <llama.cpp root> <apply|restore>
"""
import hashlib
import os
import sys

# ---------------------------------------------------------------- vecdotq.cuh
VEC_ANCHOR = """static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {"""

VEC_NEW = """// VDR 8: one call covers a whole 32-value block, so the fp32 scale is applied once per block
// instead of once per 8 values. Only reachable from the wide MMVQ path [GGML_CUDA_MMVQ_WIDE].
static __device__ __forceinline__ float vec_dot_q8_0_q8_1_wide(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq + kbx;

    int v[8];
    int u[8];

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = get_int_b2(bq8_0->qs, iqs + i);
        u[i] = get_int_b4(bq8_1->qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl<float, 8>(v, u, bq8_0->d, __low2half(bq8_1->ds));
}

""" + VEC_ANCHOR

# ---------------------------------------------------------------- mmvq.cu
MMVQ = [
    # 1) wide getters, placed AFTER the originals they call
    (
        "        default:                return 1;\n"
        "    }\n"
        "}\n"
        "\n"
        "enum mmvq_parameter_table_id {",
        "        default:                return 1;\n"
        "    }\n"
        "}\n"
        "\n"
        "// [GGML_CUDA_MMVQ_WIDE] wide variants, only implemented for Q8_0\n"
        "static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda_wide(ggml_type type) {\n"
        "    switch (type) {\n"
        "        case GGML_TYPE_Q8_0: return vec_dot_q8_0_q8_1_wide;\n"
        "        default:             return get_vec_dot_q_cuda(type);\n"
        "    }\n"
        "}\n"
        "\n"
        "static constexpr __host__ __device__ int get_vdr_mmvq_wide(ggml_type type) {\n"
        "    return type == GGML_TYPE_Q8_0 ? 8 : get_vdr_mmvq(type);\n"
        "}\n"
        "\n"
        "enum mmvq_parameter_table_id {",
    ),
    # 2) kernel template: wide flag
    (
        "template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>\n"
        "__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)\n"
        "static __global__ void mul_mat_vec_q(",
        "template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false, bool wide = false>\n"
        "__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)\n"
        "static __global__ void mul_mat_vec_q(",
    ),
    (
        "    constexpr int vdr = get_vdr_mmvq(type);\n"
        "    constexpr mmvq_parameter_table_id table_id = get_device_table_id();",
        "    constexpr int vdr = wide ? get_vdr_mmvq_wide(type) : get_vdr_mmvq(type);\n"
        "    constexpr mmvq_parameter_table_id table_id = get_device_table_id();",
    ),
    (
        "    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);\n"
        "\n"
        "    const     int tid = warp_size*threadIdx.y + threadIdx.x;",
        "    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = wide ? get_vec_dot_q_cuda_wide(type) : get_vec_dot_q_cuda(type);\n"
        "\n"
        "    const     int tid = warp_size*threadIdx.y + threadIdx.x;",
    ),
    # 3) launcher template: wide flag
    (
        "template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false>\n"
        "static void mul_mat_vec_q_switch_fusion(",
        "template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false, bool wide = false>\n"
        "static void mul_mat_vec_q_switch_fusion(",
    ),
    (
        "            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters>, launch_params,",
        "            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters, wide>, launch_params,",
    ),
    (
        "    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters>, launch_params,\n"
        "        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,",
        "    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, wide>, launch_params,\n"
        "        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,",
    ),
    # 4) env gate + dispatch for Q8_0 ncols_dst == 8
    (
        "template <ggml_type type>\n"
        "static void mul_mat_vec_q_switch_ncols_dst(",
        "// [GGML_CUDA_MMVQ_WIDE] opt-in wide (vdr = 8) Q8_0 path for the verify batch\n"
        "static bool mmvq_wide_enabled() {\n"
        "    static const bool enabled = getenv(\"GGML_CUDA_MMVQ_WIDE\") != nullptr;\n"
        "    return enabled;\n"
        "}\n"
        "\n"
        "template <ggml_type type>\n"
        "static void mul_mat_vec_q_switch_ncols_dst(",
    ),
    (
        "        case 8: {\n"
        "            constexpr int c_ncols_dst = 8;\n"
        "            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);\n"
        "            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,\n"
        "                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,\n"
        "                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,\n"
        "                 dims.first, dims.second, 0, ids_stride, stream);\n"
        "        } break;",
        "        case 8: {\n"
        "            constexpr int c_ncols_dst = 8;\n"
        "            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);\n"
        "            if constexpr (type == GGML_TYPE_Q8_0) {\n"
        "                if (mmvq_wide_enabled()) {\n"
        "                    mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, false, true>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,\n"
        "                         channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,\n"
        "                         sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,\n"
        "                         dims.first, dims.second, 0, ids_stride, stream);\n"
        "                    break;\n"
        "                }\n"
        "            }\n"
        "            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,\n"
        "                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,\n"
        "                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,\n"
        "                 dims.first, dims.second, 0, ids_stride, stream);\n"
        "        } break;",
    ),
]


def md5_of(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def patch_file(path, pairs, marker):
    with open(path, "rb") as f:
        text = f.read().decode("utf-8")
    if marker in text:
        print("  ALREADY " + os.path.basename(path))
        return
    orig = path + ".orig-wide"
    if not os.path.exists(orig):
        with open(orig, "wb") as f:
            f.write(text.encode("utf-8"))
    nl = "\r\n" if "\r\n" in text else "\n"
    for i, (old, new) in enumerate(pairs):
        old = old.replace("\n", nl)
        new = new.replace("\n", nl)
        n = text.count(old)
        if n != 1:
            sys.exit("FAIL %s anchor %d count = %d" % (os.path.basename(path), i, n))
        text = text.replace(old, new, 1)
    with open(path, "wb") as f:
        f.write(text.encode("utf-8"))
    print("  PATCHED " + os.path.basename(path) + " md5=" + md5_of(path))


def restore_file(path):
    orig = path + ".orig-wide"
    if not os.path.exists(orig):
        sys.exit("FAIL no pristine copy for " + path)
    with open(orig, "rb") as f:
        data = f.read()
    with open(path, "wb") as f:
        f.write(data)
    print("  RESTORED " + os.path.basename(path) + " md5=" + md5_of(path))


def main():
    root = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "apply"
    vec = os.path.join(root, "ggml", "src", "ggml-cuda", "vecdotq.cuh")
    mmv = os.path.join(root, "ggml", "src", "ggml-cuda", "mmvq.cu")
    if mode == "restore":
        restore_file(vec)
        restore_file(mmv)
        return
    patch_file(vec, [(VEC_ANCHOR, VEC_NEW)], "vec_dot_q8_0_q8_1_wide")
    patch_file(mmv, MMVQ, "GGML_CUDA_MMVQ_WIDE")


if __name__ == "__main__":
    main()
