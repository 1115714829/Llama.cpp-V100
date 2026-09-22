#!/usr/bin/env python3
# R224: enable MMVQ for ncols_dst 9..16 (Q8_0 only).
# Bytes-only I/O with CRLF-aware newlines, so the file's line endings are untouched.
# usage: patch-mmvq-wide.py [ROOT]
import hashlib, sys, pathlib

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '/root/llm/test/v100-opt/llama.cpp/ggml/src/ggml-cuda')
CUH = ROOT / 'mmvq.cuh'
CU  = ROOT / 'mmvq.cu'
WIDE = 16
NL = b'\r\n'
Q  = chr(34)  # double quote, keeps all literals escape-free

def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()

def edit_line(path, old, new):
    b = path.read_bytes()
    if new in b:
        print('ALREADY_APPLIED ' + path.name)
        return b
    if b.count(old) != 1:
        print('ANCHOR_MISS ' + path.name + ' found=' + str(b.count(old)))
        sys.exit(2)
    b = b.replace(old, new, 1)
    path.write_bytes(b)
    return b

print('MD5_BEFORE_CUH=' + md5(CUH))
print('MD5_BEFORE_CU=' + md5(CU))

edit_line(CUH, b'#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.',
                b'#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.' + NL +
                b'#define MMVQ_MAX_BATCH_SIZE_WIDE 16 // R224: widest MMVQ batch instantiated (Q8_0 only)')

b = edit_line(CU, b'    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);',
                  b'    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE_WIDE);')

# locate the end of the case-8 block via a unique single line, then insert after it
key = b'            constexpr int c_ncols_dst = 8;'
if b.count(key) != 1:
    print('KEY_MISS found=' + str(b.count(key)))
    sys.exit(3)
i = b.find(key)
j = b.find(b'} break;', i)
k = b.find(b'\n', j) + 1
cases = []
for n in range(9, WIDE + 1):
    cases += [
        '        case ' + str(n) + ': {',
        '            constexpr int c_ncols_dst = ' + str(n) + ';',
        '            // R224: only Q8_0 is instantiated wide, other types would multiply the kernel count by 8.',
        '            if constexpr (type == GGML_TYPE_Q8_0) {',
        '                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);',
        '                mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,',
        '                     channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,',
        '                     sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,',
        '                     dims.first, dims.second, 0, ids_stride, stream);',
        '            } else {',
        '                GGML_ABORT(' + Q + 'wide ncols_dst is only instantiated for Q8_0' + Q + ');',
        '            }',
        '        } break;',
    ]
ins = (NL.join([c.encode() for c in cases]) + NL)
if b'case 16: {' in b:
    print('ALREADY_APPLIED mmvq.cu cases')
else:
    CU.write_bytes(b[:k] + ins + b[k:])

print('MD5_AFTER_CUH=' + md5(CUH))
print('MD5_AFTER_CU=' + md5(CU))
print('APPLIED')