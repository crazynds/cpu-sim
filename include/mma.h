/* Warp matrix multiply-accumulate (nvcuda::wmma) on the CPU: the tensor core
 * API, for the 16x16x16 shape with
 *   half x half       -> float or half accumulators   (sm_70: F16C + FMA)
 *   int8 x int8       -> int accumulators, signed or unsigned (sm_72: AVX512-VNNI)
 *   bfloat16 x bf16   -> float accumulators            (sm_80: AVX512-BF16)
 * A kernel using a type the CPU has no instructions for fails to launch with
 * cudaErrorNoKernelImageForDevice, as on a GPU of an older generation; the
 * device's compute capability says which ones are there.
 *
 * A fragment holds 8 elements per thread. Its layout is unspecified in CUDA;
 * here element j of lane l is row 2j + l / 16, column l % 16 of the tile, for
 * every fragment kind (so element j of the warp is two whole rows, which the
 * native routines read directly, see src/tensor/mma.cpp). mma_sync is a warp
 * operation the compiler turns into one call of those routines per warp.
 * Not here (yet): the other shapes (m32n8k16, m8n32k16), tf32, double, the
 * sub-byte experimental types, and satf for integer accumulators. */
#ifndef CPUSIM_MMA_H
#define CPUSIM_MMA_H

#include "cuda_bf16.h"
#include "cuda_fp16.h"

extern "C" {
__device__ __attribute__((convergent)) void __cpusim_mma_f16_f32(float* d, const __half* a, const __half* b, const float* c);
__device__ __attribute__((convergent)) void __cpusim_mma_bf16_f32(float* d, const __nv_bfloat16* a, const __nv_bfloat16* b,
                                                                  const float* c);
__device__ __attribute__((convergent)) void __cpusim_mma_s8_s32(int* d, const signed char* a, const signed char* b, const int* c);
__device__ __attribute__((convergent)) void __cpusim_mma_u8_s32(int* d, const unsigned char* a, const unsigned char* b,
                                                                const int* c);
// Element j of D for the calling lane, B and C by value, A by pointer. The
// compiler turns the 8 calls of one mma into one inline mma; the accumulator
// and B can stay in registers, and A's pairs are broadcast from its fragment
// in memory. On the CPU it pays to load A one step of k ahead of its mma.
#define __CPUSIM_MMAV_ARGS(T) T, T, T, T, T, T, T, T
__device__ __attribute__((convergent, pure)) float __cpusim_mmav_bf16_f32(int j, const __nv_bfloat16* a,
                                                                          __CPUSIM_MMAV_ARGS(unsigned short),
                                                                          __CPUSIM_MMAV_ARGS(float));
__device__ __attribute__((convergent, pure)) float __cpusim_mmav_f16_f32(int j, const __half* a,
                                                                         __CPUSIM_MMAV_ARGS(unsigned short),
                                                                         __CPUSIM_MMAV_ARGS(float));
__device__ __attribute__((convergent, pure)) int __cpusim_mmav_s8_s32(int j, const signed char* a,
                                                                     __CPUSIM_MMAV_ARGS(signed char),
                                                                     __CPUSIM_MMAV_ARGS(int));
__device__ __attribute__((convergent, pure)) int __cpusim_mmav_u8_s32(int j, const unsigned char* a,
                                                                     __CPUSIM_MMAV_ARGS(unsigned char),
                                                                     __CPUSIM_MMAV_ARGS(int));
#undef __CPUSIM_MMAV_ARGS
}

namespace nvcuda {
namespace wmma {

struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};
enum layout_t { mem_row_major, mem_col_major };

template <typename T, int size>
struct __frag_base {
    enum { num_elements = size };
    typedef T element_type;
    typedef T storage_element_type;
    T x[size];
};

template <typename Use, int m, int n, int k, typename T, typename Layout = void>
class fragment;  // only the shapes and types listed above exist

#define __CPUSIM_FRAG(USE, T)                                                                 \
    template <typename Layout>                                                                \
    class fragment<USE, 16, 16, 16, T, Layout> : public __frag_base<T, 8> {};
__CPUSIM_FRAG(matrix_a, __half)
__CPUSIM_FRAG(matrix_b, __half)
__CPUSIM_FRAG(matrix_a, __nv_bfloat16)
__CPUSIM_FRAG(matrix_b, __nv_bfloat16)
__CPUSIM_FRAG(matrix_a, signed char)
__CPUSIM_FRAG(matrix_b, signed char)
__CPUSIM_FRAG(matrix_a, unsigned char)
__CPUSIM_FRAG(matrix_b, unsigned char)
__CPUSIM_FRAG(accumulator, float)
__CPUSIM_FRAG(accumulator, __half)
__CPUSIM_FRAG(accumulator, int)
#undef __CPUSIM_FRAG

namespace __cpusim {
template <typename T>
struct identity {
    typedef T type;
};
// Row and column of element j of the calling lane.
static __device__ __forceinline__ unsigned row(int j) { return 2u * (unsigned)j + (__cpusim_laneid() >> 4); }
static __device__ __forceinline__ unsigned col() { return __cpusim_laneid() & 15u; }
}  // namespace __cpusim

template <typename Use, int m, int n, int k, typename T, typename Layout>
__device__ __forceinline__ void fill_fragment(fragment<Use, m, n, k, T, Layout>& f,
                                              const typename __cpusim::identity<T>::type& v) {
#pragma unroll
    for (int j = 0; j < 8; ++j) f.x[j] = v;
}

// A and B: the layout is part of the fragment type. A is rows x k, B is k x columns.
template <typename Use, typename T>
__device__ __forceinline__ void load_matrix_sync(fragment<Use, 16, 16, 16, T, row_major>& f, const T* p, unsigned ldm) {
#pragma unroll
    for (int j = 0; j < 8; ++j) f.x[j] = p[__cpusim::row(j) * ldm + __cpusim::col()];
}
template <typename Use, typename T>
__device__ __forceinline__ void load_matrix_sync(fragment<Use, 16, 16, 16, T, col_major>& f, const T* p, unsigned ldm) {
#pragma unroll
    for (int j = 0; j < 8; ++j) f.x[j] = p[__cpusim::col() * ldm + __cpusim::row(j)];
}

// Accumulators: the layout is an argument.
template <typename T>
__device__ __forceinline__ void load_matrix_sync(fragment<accumulator, 16, 16, 16, T>& f, const T* p, unsigned ldm,
                                                 layout_t layout) {
    if (layout == mem_row_major) {
#pragma unroll
        for (int j = 0; j < 8; ++j) f.x[j] = p[__cpusim::row(j) * ldm + __cpusim::col()];
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) f.x[j] = p[__cpusim::col() * ldm + __cpusim::row(j)];
    }
}
template <typename T>
__device__ __forceinline__ void store_matrix_sync(T* p, const fragment<accumulator, 16, 16, 16, T>& f, unsigned ldm,
                                                  layout_t layout) {
    if (layout == mem_row_major) {
#pragma unroll
        for (int j = 0; j < 8; ++j) p[__cpusim::row(j) * ldm + __cpusim::col()] = f.x[j];
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) p[__cpusim::col() * ldm + __cpusim::row(j)] = f.x[j];
    }
}

namespace __cpusim {
// satf for floating point accumulators: infinities to the largest finite value, NaN to 0.
static __device__ __forceinline__ void saturate(fragment<accumulator, 16, 16, 16, float>& d) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float v = d.x[j];
        d.x[j] = v != v ? 0.0f : v > 3.402823466e38f ? 3.402823466e38f : v < -3.402823466e38f ? -3.402823466e38f : v;
    }
}
}  // namespace __cpusim

// The 8 elements of a fragment as arguments (with a member access appended).
#define __CPUSIM_E(f, s) f.x[0] s, f.x[1] s, f.x[2] s, f.x[3] s, f.x[4] s, f.x[5] s, f.x[6] s, f.x[7] s

// D = A * B + C. The fragments' layouts (row/col major) only affected how they
// were loaded; the tiles are the same.
template <typename LA, typename LB>
__device__ __forceinline__ void mma_sync(fragment<accumulator, 16, 16, 16, float>& d,
                                         const fragment<matrix_a, 16, 16, 16, __half, LA>& a,
                                         const fragment<matrix_b, 16, 16, 16, __half, LB>& b,
                                         const fragment<accumulator, 16, 16, 16, float>& c, bool satf = false) {
    float r[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) r[j] = __cpusim_mmav_f16_f32(j, a.x, __CPUSIM_E(b, .__x), __CPUSIM_E(c, ));
#pragma unroll
    for (int j = 0; j < 8; ++j) d.x[j] = r[j];
    if (satf) __cpusim::saturate(d);
}
template <typename LA, typename LB>
__device__ __forceinline__ void mma_sync(fragment<accumulator, 16, 16, 16, float>& d,
                                         const fragment<matrix_a, 16, 16, 16, __nv_bfloat16, LA>& a,
                                         const fragment<matrix_b, 16, 16, 16, __nv_bfloat16, LB>& b,
                                         const fragment<accumulator, 16, 16, 16, float>& c, bool satf = false) {
    float r[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) r[j] = __cpusim_mmav_bf16_f32(j, a.x, __CPUSIM_E(b, .__x), __CPUSIM_E(c, ));
#pragma unroll
    for (int j = 0; j < 8; ++j) d.x[j] = r[j];
    if (satf) __cpusim::saturate(d);
}
// Half accumulators (as either input or output): computed in float, rounded once.
template <typename TD, typename TC, typename LA, typename LB>
__device__ __forceinline__ void mma_sync(fragment<accumulator, 16, 16, 16, TD>& d,
                                         const fragment<matrix_a, 16, 16, 16, __half, LA>& a,
                                         const fragment<matrix_b, 16, 16, 16, __half, LB>& b,
                                         const fragment<accumulator, 16, 16, 16, TC>& c, bool satf = false) {
    fragment<accumulator, 16, 16, 16, float> t;
#pragma unroll
    for (int j = 0; j < 8; ++j) t.x[j] = (float)c.x[j];
    float r[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) r[j] = __cpusim_mmav_f16_f32(j, a.x, __CPUSIM_E(b, .__x), __CPUSIM_E(t, ));
#pragma unroll
    for (int j = 0; j < 8; ++j) t.x[j] = r[j];
    if (satf) __cpusim::saturate(t);
#pragma unroll
    for (int j = 0; j < 8; ++j) d.x[j] = (TD)t.x[j];
}
template <typename LA, typename LB>
__device__ __forceinline__ void mma_sync(fragment<accumulator, 16, 16, 16, int>& d,
                                         const fragment<matrix_a, 16, 16, 16, signed char, LA>& a,
                                         const fragment<matrix_b, 16, 16, 16, signed char, LB>& b,
                                         const fragment<accumulator, 16, 16, 16, int>& c, bool satf = false) {
    int r[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) r[j] = __cpusim_mmav_s8_s32(j, a.x, __CPUSIM_E(b, ), __CPUSIM_E(c, ));
#pragma unroll
    for (int j = 0; j < 8; ++j) d.x[j] = r[j];
}
template <typename LA, typename LB>
__device__ __forceinline__ void mma_sync(fragment<accumulator, 16, 16, 16, int>& d,
                                         const fragment<matrix_a, 16, 16, 16, unsigned char, LA>& a,
                                         const fragment<matrix_b, 16, 16, 16, unsigned char, LB>& b,
                                         const fragment<accumulator, 16, 16, 16, int>& c, bool satf = false) {
    int r[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) r[j] = __cpusim_mmav_u8_s32(j, a.x, __CPUSIM_E(b, ), __CPUSIM_E(c, ));
#pragma unroll
    for (int j = 0; j < 8; ++j) d.x[j] = r[j];
}

#undef __CPUSIM_E

}  // namespace wmma
}  // namespace nvcuda

#endif
