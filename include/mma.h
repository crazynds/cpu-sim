/* Warp matrix multiply-accumulate (nvcuda::wmma) on the CPU: the tensor core
 * API with
 *   half x half       -> float or half accumulators, 16x16x16  (sm_70: F16C + FMA)
 *   int8 x int8       -> int, signed or unsigned, 16x16x16     (sm_72: AVX512-VNNI)
 *   bfloat16 x bf16   -> float, 16x16x16                       (sm_80: AVX512-BF16)
 *   tf32 x tf32       -> float, 16x16x8                        (sm_80: FMA)
 *   double x double   -> double, 8x8x4                         (sm_80: FMA)
 * A kernel using a type the CPU has no instructions for fails to launch with
 * cudaErrorNoKernelImageForDevice, as on a GPU of an older generation; the
 * device's compute capability says which ones are there.
 *
 * Fragment layout (unspecified in CUDA). For a fragment whose tile rows have
 * C columns (A: k, B: n, the accumulator: n), element j of lane l is
 *     row (32 / C) * j + l / C,  column l % C,
 * so element j over a warp's 32 lanes is 32 / C whole rows of the tile, one
 * after the other: the compiler's inline mma reads B's and C's rows straight
 * out of registers and broadcasts A's values from its fragment in memory.
 * mma_sync becomes that inline code, one per warp (src/compiler/simt.cpp).
 * Not here (yet): the shapes m32n8k16 and m8n32k16, the sub-byte
 * experimental types, and satf for integer accumulators. */
#ifndef CPUSIM_MMA_H
#define CPUSIM_MMA_H

#include "cuda_bf16.h"
#include "cuda_fp16.h"

extern "C" {
// Element j of D for the calling lane: A by pointer (its fragment), B's and
// C's elements by value. The compiler turns the calls of one mma_sync into one
// inline mma; the accumulator and B can stay in registers, and A's values are
// broadcast from its fragment in memory. On the CPU it pays to load A one step
// of k ahead of its mma.
#define __CPUSIM_X8(T) T, T, T, T, T, T, T, T
#define __CPUSIM_X4(T) T, T, T, T
#define __CPUSIM_MMAV(R, NAME, A, B, C) __device__ __attribute__((convergent, pure)) R __cpusim_mmav_##NAME(int j, const A* a, B, C)
__CPUSIM_MMAV(float, bf16_f32, __nv_bfloat16, __CPUSIM_X8(unsigned short), __CPUSIM_X8(float));
__CPUSIM_MMAV(float, f16_f32, __half, __CPUSIM_X8(unsigned short), __CPUSIM_X8(float));
__CPUSIM_MMAV(int, s8_s32, signed char, __CPUSIM_X8(signed char), __CPUSIM_X8(int));
__CPUSIM_MMAV(int, u8_s32, unsigned char, __CPUSIM_X8(unsigned char), __CPUSIM_X8(int));
__CPUSIM_MMAV(float, tf32_f32, float, __CPUSIM_X4(float), __CPUSIM_X8(float));
__device__ __attribute__((convergent, pure)) double __cpusim_mmav_f64_f64(int j, const double* a, double b, double c0,
                                                                          double c1);
#undef __CPUSIM_MMAV
#undef __CPUSIM_X4
#undef __CPUSIM_X8
}

namespace nvcuda {
namespace wmma {

struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};
enum layout_t { mem_row_major, mem_col_major };
namespace precision {
struct tf32 {};  // fp32 values used with a 10-bit mantissa; fragments hold floats
}  // namespace precision

namespace __cpusim {
template <typename T>
struct identity {
    typedef T type;
};
// What a fragment of element type T holds.
template <typename T>
struct storage {
    typedef T type;
};
template <>
struct storage<precision::tf32> {
    typedef float type;
};
// Rows and columns of the tile a fragment holds.
template <typename Use, int m, int n, int k>
struct tile;
template <int m, int n, int k>
struct tile<matrix_a, m, n, k> {
    enum { rows = m, cols = k };
};
template <int m, int n, int k>
struct tile<matrix_b, m, n, k> {
    enum { rows = k, cols = n };
};
template <int m, int n, int k>
struct tile<accumulator, m, n, k> {
    enum { rows = m, cols = n };
};
// Row and column of element j of the calling lane, for tiles of `cols` columns.
template <int cols>
static __device__ __forceinline__ unsigned row(int j) {
    return (32u / cols) * (unsigned)j + __cpusim_laneid() / cols;
}
template <int cols>
static __device__ __forceinline__ unsigned col() {
    return __cpusim_laneid() % cols;
}
}  // namespace __cpusim

template <typename T, int size>
struct __frag_base {
    enum { num_elements = size };
    typedef typename __cpusim::storage<T>::type element_type;
    typedef element_type storage_element_type;
    element_type x[size];
};

template <typename Use, int m, int n, int k, typename T, typename Layout = void>
class fragment;  // only the shapes and types listed above exist

#define __CPUSIM_FRAG(USE, M, N, K, T)                                                        \
    template <typename Layout>                                                                \
    class fragment<USE, M, N, K, T, Layout>                                                   \
        : public __frag_base<T, __cpusim::tile<USE, M, N, K>::rows * __cpusim::tile<USE, M, N, K>::cols / 32> {};
__CPUSIM_FRAG(matrix_a, 16, 16, 16, __half)
__CPUSIM_FRAG(matrix_b, 16, 16, 16, __half)
__CPUSIM_FRAG(matrix_a, 16, 16, 16, __nv_bfloat16)
__CPUSIM_FRAG(matrix_b, 16, 16, 16, __nv_bfloat16)
__CPUSIM_FRAG(matrix_a, 16, 16, 16, signed char)
__CPUSIM_FRAG(matrix_b, 16, 16, 16, signed char)
__CPUSIM_FRAG(matrix_a, 16, 16, 16, unsigned char)
__CPUSIM_FRAG(matrix_b, 16, 16, 16, unsigned char)
__CPUSIM_FRAG(accumulator, 16, 16, 16, float)
__CPUSIM_FRAG(accumulator, 16, 16, 16, __half)
__CPUSIM_FRAG(accumulator, 16, 16, 16, int)
__CPUSIM_FRAG(matrix_a, 16, 16, 8, precision::tf32)
__CPUSIM_FRAG(matrix_b, 16, 16, 8, precision::tf32)
__CPUSIM_FRAG(accumulator, 16, 16, 8, float)
__CPUSIM_FRAG(matrix_a, 8, 8, 4, double)
__CPUSIM_FRAG(matrix_b, 8, 8, 4, double)
__CPUSIM_FRAG(accumulator, 8, 8, 4, double)
#undef __CPUSIM_FRAG

template <typename Use, int m, int n, int k, typename T, typename Layout>
__device__ __forceinline__ void fill_fragment(fragment<Use, m, n, k, T, Layout>& f,
                                              const typename __cpusim::identity<typename __cpusim::storage<T>::type>::type& v) {
#pragma unroll
    for (int j = 0; j < f.num_elements; ++j) f.x[j] = v;
}

// A and B: the layout is part of the fragment type. A is rows x k, B is k x columns.
template <typename Use, int m, int n, int k, typename T>
__device__ __forceinline__ void load_matrix_sync(fragment<Use, m, n, k, T, row_major>& f,
                                                 const typename __cpusim::storage<T>::type* p, unsigned ldm) {
    constexpr int C = __cpusim::tile<Use, m, n, k>::cols;
#pragma unroll
    for (int j = 0; j < f.num_elements; ++j) f.x[j] = p[__cpusim::row<C>(j) * ldm + __cpusim::col<C>()];
}
template <typename Use, int m, int n, int k, typename T>
__device__ __forceinline__ void load_matrix_sync(fragment<Use, m, n, k, T, col_major>& f,
                                                 const typename __cpusim::storage<T>::type* p, unsigned ldm) {
    constexpr int C = __cpusim::tile<Use, m, n, k>::cols;
#pragma unroll
    for (int j = 0; j < f.num_elements; ++j) f.x[j] = p[__cpusim::col<C>() * ldm + __cpusim::row<C>(j)];
}

// Accumulators: the layout is an argument.
template <int m, int n, int k, typename T>
__device__ __forceinline__ void load_matrix_sync(fragment<accumulator, m, n, k, T>& f, const T* p, unsigned ldm,
                                                 layout_t layout) {
    if (layout == mem_row_major) {
#pragma unroll
        for (int j = 0; j < f.num_elements; ++j) f.x[j] = p[__cpusim::row<n>(j) * ldm + __cpusim::col<n>()];
    } else {
#pragma unroll
        for (int j = 0; j < f.num_elements; ++j) f.x[j] = p[__cpusim::col<n>() * ldm + __cpusim::row<n>(j)];
    }
}
template <int m, int n, int k, typename T>
__device__ __forceinline__ void store_matrix_sync(T* p, const fragment<accumulator, m, n, k, T>& f, unsigned ldm,
                                                  layout_t layout) {
    if (layout == mem_row_major) {
#pragma unroll
        for (int j = 0; j < f.num_elements; ++j) p[__cpusim::row<n>(j) * ldm + __cpusim::col<n>()] = f.x[j];
    } else {
#pragma unroll
        for (int j = 0; j < f.num_elements; ++j) p[__cpusim::col<n>() * ldm + __cpusim::row<n>(j)] = f.x[j];
    }
}

// fp32 to tf32: round to nearest with ties away from zero to a 10-bit
// mantissa (cvt.rna.tf32.f32); infinities and NaN unchanged. The tensor core
// itself ignores the 13 low mantissa bits of what it is given.
static __device__ __forceinline__ float __float_to_tf32(float in) {
    unsigned u = __float_as_uint(in);
    if ((u & 0x7f800000u) == 0x7f800000u) return in;
    return __uint_as_float((u + 0x1000u) & 0xffffe000u);
}

namespace __cpusim {
// satf for floating point accumulators: infinities to the largest finite value, NaN to 0.
template <int m, int n, int k>
static __device__ __forceinline__ void saturate(fragment<accumulator, m, n, k, float>& d) {
#pragma unroll
    for (int j = 0; j < d.num_elements; ++j) {
        float v = d.x[j];
        d.x[j] = v != v ? 0.0f : v > 3.402823466e38f ? 3.402823466e38f : v < -3.402823466e38f ? -3.402823466e38f : v;
    }
}
static __device__ __forceinline__ void saturate(fragment<accumulator, 8, 8, 4, double>& d) {
#pragma unroll
    for (int j = 0; j < d.num_elements; ++j) {
        double v = d.x[j];
        d.x[j] = v != v ? 0.0 : v > 1.7976931348623157e308 ? 1.7976931348623157e308
                                 : v < -1.7976931348623157e308 ? -1.7976931348623157e308 : v;
    }
}
}  // namespace __cpusim

// The elements of a fragment as arguments (with a member access appended).
#define __CPUSIM_E8(f, s) f.x[0] s, f.x[1] s, f.x[2] s, f.x[3] s, f.x[4] s, f.x[5] s, f.x[6] s, f.x[7] s
#define __CPUSIM_E4(f, s) f.x[0] s, f.x[1] s, f.x[2] s, f.x[3] s
// D = A * B + C through __cpusim_mmav_<NAME>, element by element (N of them).
#define __CPUSIM_MMA(NAME, R, N, BARGS, CARGS)                                                \
    {                                                                                         \
        R r[N];                                                                               \
        _Pragma("unroll") for (int j = 0; j < N; ++j) r[j] = __cpusim_mmav_##NAME(j, a.x, BARGS, CARGS); \
        _Pragma("unroll") for (int j = 0; j < N; ++j) d.x[j] = r[j];                          \
    }

// The fragments' layouts (row/col major) only affected how they were loaded;
// the tiles are the same.
template <typename LA, typename LB>
__device__ __forceinline__ void mma_sync(fragment<accumulator, 16, 16, 16, float>& d,
                                         const fragment<matrix_a, 16, 16, 16, __half, LA>& a,
                                         const fragment<matrix_b, 16, 16, 16, __half, LB>& b,
                                         const fragment<accumulator, 16, 16, 16, float>& c, bool satf = false) {
    __CPUSIM_MMA(f16_f32, float, 8, __CPUSIM_E8(b, .__x), __CPUSIM_E8(c, ))
    if (satf) __cpusim::saturate(d);
}
template <typename LA, typename LB>
__device__ __forceinline__ void mma_sync(fragment<accumulator, 16, 16, 16, float>& d,
                                         const fragment<matrix_a, 16, 16, 16, __nv_bfloat16, LA>& a,
                                         const fragment<matrix_b, 16, 16, 16, __nv_bfloat16, LB>& b,
                                         const fragment<accumulator, 16, 16, 16, float>& c, bool satf = false) {
    __CPUSIM_MMA(bf16_f32, float, 8, __CPUSIM_E8(b, .__x), __CPUSIM_E8(c, ))
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
    for (int j = 0; j < 8; ++j) r[j] = __cpusim_mmav_f16_f32(j, a.x, __CPUSIM_E8(b, .__x), __CPUSIM_E8(t, ));
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
    __CPUSIM_MMA(s8_s32, int, 8, __CPUSIM_E8(b, ), __CPUSIM_E8(c, ))
}
template <typename LA, typename LB>
__device__ __forceinline__ void mma_sync(fragment<accumulator, 16, 16, 16, int>& d,
                                         const fragment<matrix_a, 16, 16, 16, unsigned char, LA>& a,
                                         const fragment<matrix_b, 16, 16, 16, unsigned char, LB>& b,
                                         const fragment<accumulator, 16, 16, 16, int>& c, bool satf = false) {
    __CPUSIM_MMA(u8_s32, int, 8, __CPUSIM_E8(b, ), __CPUSIM_E8(c, ))
}
template <typename LA, typename LB>
__device__ __forceinline__ void mma_sync(fragment<accumulator, 16, 16, 8, float>& d,
                                         const fragment<matrix_a, 16, 16, 8, precision::tf32, LA>& a,
                                         const fragment<matrix_b, 16, 16, 8, precision::tf32, LB>& b,
                                         const fragment<accumulator, 16, 16, 8, float>& c, bool satf = false) {
    __CPUSIM_MMA(tf32_f32, float, 8, __CPUSIM_E4(b, ), __CPUSIM_E8(c, ))
    if (satf) __cpusim::saturate(d);
}
template <typename LA, typename LB>
__device__ __forceinline__ void mma_sync(fragment<accumulator, 8, 8, 4, double>& d,
                                         const fragment<matrix_a, 8, 8, 4, double, LA>& a,
                                         const fragment<matrix_b, 8, 8, 4, double, LB>& b,
                                         const fragment<accumulator, 8, 8, 4, double>& c, bool satf = false) {
    const double r0 = __cpusim_mmav_f64_f64(0, a.x, b.x[0], c.x[0], c.x[1]);
    const double r1 = __cpusim_mmav_f64_f64(1, a.x, b.x[0], c.x[0], c.x[1]);
    d.x[0] = r0;
    d.x[1] = r1;
    if (satf) __cpusim::saturate(d);
}

#undef __CPUSIM_MMA
#undef __CPUSIM_E4
#undef __CPUSIM_E8

}  // namespace wmma
}  // namespace nvcuda

#endif
