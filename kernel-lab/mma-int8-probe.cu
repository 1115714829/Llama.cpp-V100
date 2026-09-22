// Does sm_70 support an int8 mma? (If yes, the Q8_0 codes could feed the tensor cores raw,
// with no int8->f16 decode at all, which is what currently eats the mma path's advantage.)
#include <cstdint>

__global__ void probe_m8n8k4_s8(const int8_t * a, const int8_t * b, int32_t * d) {
    int32_t  D[2] = {0, 0};
    uint32_t A    = *reinterpret_cast<const uint32_t *>(a);
    uint32_t B    = *reinterpret_cast<const uint32_t *>(b);
    asm("mma.sync.aligned.m8n8k4.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
        : "+r"(D[0]), "+r"(D[1])
        : "r"(A), "r"(B));
    d[0] = D[0];
}

__global__ void probe_m16n8k16_s8(const int8_t * a, const int8_t * b, int32_t * d) {
    int32_t  D[4] = {0, 0, 0, 0};
    uint32_t A[4] = {0, 0, 0, 0};
    uint32_t B[2] = {0, 0};
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
    d[0] = D[0];
}
