/* Warp matrix multiply-accumulate (nvcuda::wmma) on the CPU: the tensor core
 * API with
 *   half x half       -> float or half accumulators  (sm_70: F16C + FMA)
 *   int8 x int8       -> int, signed or unsigned     (sm_72: AVX512-VNNI)
 *   bfloat16 x bf16   -> float                       (sm_80: AVX512-BF16)
 * each in the shapes 16x16x16, 32x8x16 and 8x32x16, and
 *   tf32 x tf32       -> float, 16x16x8                        (sm_80: FMA)
 *   double x double   -> double, 8x8x4                         (sm_80: FMA)
 *   experimental: s4, u4 -> int, 8x8x32; b1 (xor/and, popc) -> int, 8x8x128
 *                                                   (sm_75: AVX512-VNNI, VPOPCNTDQ)
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
 * Not here: satf for integer accumulators (ignored). */
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
// m32n8k16 (B: 4 elements per lane) and m8n32k16 (B: 16).
#define __CPUSIM_X16(T) __CPUSIM_X8(T), __CPUSIM_X8(T)
__CPUSIM_MMAV(float, bf16_f32_m32n8k16, __nv_bfloat16, __CPUSIM_X4(unsigned short), __CPUSIM_X8(float));
__CPUSIM_MMAV(float, f16_f32_m32n8k16, __half, __CPUSIM_X4(unsigned short), __CPUSIM_X8(float));
__CPUSIM_MMAV(int, s8_s32_m32n8k16, signed char, __CPUSIM_X4(signed char), __CPUSIM_X8(int));
__CPUSIM_MMAV(int, u8_s32_m32n8k16, unsigned char, __CPUSIM_X4(unsigned char), __CPUSIM_X8(int));
__CPUSIM_MMAV(float, bf16_f32_m8n32k16, __nv_bfloat16, __CPUSIM_X16(unsigned short), __CPUSIM_X8(float));
__CPUSIM_MMAV(float, f16_f32_m8n32k16, __half, __CPUSIM_X16(unsigned short), __CPUSIM_X8(float));
__CPUSIM_MMAV(int, s8_s32_m8n32k16, signed char, __CPUSIM_X16(signed char), __CPUSIM_X8(int));
__CPUSIM_MMAV(int, u8_s32_m8n32k16, unsigned char, __CPUSIM_X16(unsigned char), __CPUSIM_X8(int));
#undef __CPUSIM_X16
// Sub-byte types: A's and B's 32 bits per lane by value/pointer, C's 2 elements.
__device__ __attribute__((convergent, pure)) int __cpusim_mmav_s4_s32(int j, const unsigned* a, unsigned b, int c0, int c1);
__device__ __attribute__((convergent, pure)) int __cpusim_mmav_u4_s32(int j, const unsigned* a, unsigned b, int c0, int c1);
__device__ __attribute__((convergent, pure)) int __cpusim_mmav_b1xor_s32(int j, const unsigned* a, unsigned b, int c0,
                                                                        int c1);
__device__ __attribute__((convergent, pure)) int __cpusim_mmav_b1and_s32(int j, const unsigned* a, unsigned b, int c0,
                                                                        int c1);
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
namespace experimental {
namespace precision {
struct u4 {};  // 4-bit integers, 8 to a 32-bit word
struct s4 {};
struct b1 {};  // bits, 32 to a word
}  // namespace precision
enum bmmaBitOp { bmmaBitOpXOR = 1, bmmaBitOpAND = 2 };
enum bmmaAccumulateOp { bmmaAccumulateOpPOPC = 1 };
}  // namespace experimental

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
// Offset of element j in a row-major matrix with leading dimension ldm.
template <typename Use, int m, int n, int k>
static __device__ __forceinline__ unsigned at(int j, unsigned ldm) {
    typedef tile<Use, m, n, k> t;
    return row<t::cols>(j) * ldm + col<t::cols>();
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
#define __CPUSIM_FRAGS(M, N, K)                 \
    __CPUSIM_FRAG(matrix_a, M, N, K, __half)        \
    __CPUSIM_FRAG(matrix_b, M, N, K, __half)        \
    __CPUSIM_FRAG(matrix_a, M, N, K, __nv_bfloat16) \
    __CPUSIM_FRAG(matrix_b, M, N, K, __nv_bfloat16) \
    __CPUSIM_FRAG(matrix_a, M, N, K, signed char)   \
    __CPUSIM_FRAG(matrix_b, M, N, K, signed char)   \
    __CPUSIM_FRAG(matrix_a, M, N, K, unsigned char) \
    __CPUSIM_FRAG(matrix_b, M, N, K, unsigned char) \
    __CPUSIM_FRAG(accumulator, M, N, K, float)      \
    __CPUSIM_FRAG(accumulator, M, N, K, __half)     \
    __CPUSIM_FRAG(accumulator, M, N, K, int)
__CPUSIM_FRAGS(32, 8, 16)
__CPUSIM_FRAGS(8, 32, 16)
#undef __CPUSIM_FRAGS
__CPUSIM_FRAG(matrix_a, 16, 16, 8, precision::tf32)
__CPUSIM_FRAG(matrix_b, 16, 16, 8, precision::tf32)
__CPUSIM_FRAG(accumulator, 16, 16, 8, float)
__CPUSIM_FRAG(matrix_a, 8, 8, 4, double)
__CPUSIM_FRAG(matrix_b, 8, 8, 4, double)
__CPUSIM_FRAG(accumulator, 8, 8, 4, double)
__CPUSIM_FRAG(accumulator, 8, 8, 32, int)
__CPUSIM_FRAG(accumulator, 8, 8, 128, int)
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
#pragma unroll
    for (int j = 0; j < f.num_elements; ++j) f.x[j] = p[__cpusim::at<Use, m, n, k>(j, ldm)];
}
// Column-major: the transpose's row-major offsets.
template <typename Use, int m, int n, int k, typename T>
__device__ __forceinline__ void load_matrix_sync(fragment<Use, m, n, k, T, col_major>& f,
                                                 const typename __cpusim::storage<T>::type* p, unsigned ldm) {
    typedef __cpusim::tile<Use, m, n, k> t;
#pragma unroll
    for (int j = 0; j < f.num_elements; ++j) {
        const unsigned r = __cpusim::row<t::cols>(j), c = __cpusim::col<t::cols>();
        f.x[j] = p[c * ldm + r];
    }
}

// Accumulators: the layout is an argument.
namespace __cpusim {
// Offset of accumulator element j in memory of the given layout.
template <int m, int n, int k>
static __device__ __forceinline__ unsigned acc_at(int j, unsigned ldm, layout_t layout) {
    typedef tile<accumulator, m, n, k> t;
    const unsigned r = row<t::cols>(j), c = col<t::cols>();
    return layout == mem_row_major ? r * ldm + c : c * ldm + r;
}
}  // namespace __cpusim
template <int m, int n, int k, typename T>
__device__ __forceinline__ void load_matrix_sync(fragment<accumulator, m, n, k, T>& f, const T* p, unsigned ldm,
                                                 layout_t layout) {
#pragma unroll
    for (int j = 0; j < f.num_elements; ++j) f.x[j] = p[__cpusim::acc_at<m, n, k>(j, ldm, layout)];
}
template <int m, int n, int k, typename T>
__device__ __forceinline__ void store_matrix_sync(T* p, const fragment<accumulator, m, n, k, T>& f, unsigned ldm,
                                                  layout_t layout) {
#pragma unroll
    for (int j = 0; j < f.num_elements; ++j) p[__cpusim::acc_at<m, n, k>(j, ldm, layout)] = f.x[j];
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
#define __CPUSIM_E4(f, s) f.x[0] s, f.x[1] s, f.x[2] s, f.x[3] s
#define __CPUSIM_E8(f, s) __CPUSIM_E4(f, s), f.x[4] s, f.x[5] s, f.x[6] s, f.x[7] s
#define __CPUSIM_E16(f, s) \
    __CPUSIM_E8(f, s), f.x[8] s, f.x[9] s, f.x[10] s, f.x[11] s, f.x[12] s, f.x[13] s, f.x[14] s, f.x[15] s
// D = A * B + C for one shape and type pair: the 8 elements of D through
// __cpusim_mmav_<NAME> (NB elements of B, member BS: .__x for 16-bit types).
// The fragments' layouts (row/col major) only affected how they were loaded;
// the tiles are the same.
#define __CPUSIM_MMA(M, N, K, TI, TC, NAME, NB, BS, SATF, A, B)                               \
    template <typename LA, typename LB>                                                       \
    __device__ __forceinline__ void mma_sync(fragment<accumulator, M, N, K, TC>& d,           \
                                             const fragment<matrix_a, M, N, K, TI, LA>& a,    \
                                             const fragment<matrix_b, M, N, K, TI, LB>& b,    \
                                             const fragment<accumulator, M, N, K, TC>& c, bool satf = false) { \
        TC r[8];                                                                              \
        _Pragma("unroll") for (int j = 0; j < 8; ++j) r[j] = __cpusim_mmav_##NAME(j, A.x, __CPUSIM_E##NB(B, BS), __CPUSIM_E8(c, )); \
        _Pragma("unroll") for (int j = 0; j < 8; ++j) d.x[j] = r[j];                          \
        SATF                                                                                  \
    }
#define __CPUSIM_SATF if (satf) __cpusim::saturate(d);
#define __CPUSIM_MMAS(M, N, K, SUFFIX, NB, A, B)                                              \
    __CPUSIM_MMA(M, N, K, __half, float, f16_f32##SUFFIX, NB, .__x, __CPUSIM_SATF, A, B)      \
    __CPUSIM_MMA(M, N, K, __nv_bfloat16, float, bf16_f32##SUFFIX, NB, .__x, __CPUSIM_SATF, A, B) \
    __CPUSIM_MMA(M, N, K, signed char, int, s8_s32##SUFFIX, NB, , , A, B)                     \
    __CPUSIM_MMA(M, N, K, unsigned char, int, u8_s32##SUFFIX, NB, , , A, B)
__CPUSIM_MMAS(16, 16, 16, , 8, a, b)
__CPUSIM_MMAS(8, 32, 16, _m8n32k16, 16, a, b)
__CPUSIM_MMAS(32, 8, 16, _m32n8k16, 4, a, b)
__CPUSIM_MMA(16, 16, 8, precision::tf32, float, tf32_f32, 4, , __CPUSIM_SATF, a, b)
#undef __CPUSIM_MMAS
#undef __CPUSIM_SATF

// Half accumulators (as either input or output), any shape: computed in
// float, rounded once.
template <int m, int n, int k, typename TD, typename TC, typename LA, typename LB>
__device__ __forceinline__ void mma_sync(fragment<accumulator, m, n, k, TD>& d,
                                         const fragment<matrix_a, m, n, k, __half, LA>& a,
                                         const fragment<matrix_b, m, n, k, __half, LB>& b,
                                         const fragment<accumulator, m, n, k, TC>& c, bool satf = false) {
    fragment<accumulator, m, n, k, float> t;
#pragma unroll
    for (int j = 0; j < t.num_elements; ++j) t.x[j] = (float)c.x[j];
    mma_sync(t, a, b, t, satf);
#pragma unroll
    for (int j = 0; j < t.num_elements; ++j) d.x[j] = (TD)t.x[j];
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

// Sub-byte types (experimental in CUDA too): A row-major, B column-major,
// one 32-bit word per lane. Lane l holds word l % 4 of row (A) or column (B)
// l / 4: rows and columns are 4 words, 32 values of 4 bits (8x8x32) or 128
// bits (8x8x128). ldm counts elements (4-bit values or bits), as in CUDA.
namespace __cpusim {
template <typename T>
struct subbyte;
template <>
struct subbyte<experimental::precision::s4> {
    enum { k = 32, bits = 4 };
};
template <>
struct subbyte<experimental::precision::u4> {
    enum { k = 32, bits = 4 };
};
template <>
struct subbyte<experimental::precision::b1> {
    enum { k = 128, bits = 1 };
};
}  // namespace __cpusim
#define __CPUSIM_SUBFRAG(USE, T, L)                                                            \
    template <>                                                                                \
    class fragment<USE, 8, 8, __cpusim::subbyte<T>::k, T, L> : public __frag_base<unsigned, 1> {};
__CPUSIM_SUBFRAG(matrix_a, experimental::precision::s4, row_major)
__CPUSIM_SUBFRAG(matrix_b, experimental::precision::s4, col_major)
__CPUSIM_SUBFRAG(matrix_a, experimental::precision::u4, row_major)
__CPUSIM_SUBFRAG(matrix_b, experimental::precision::u4, col_major)
__CPUSIM_SUBFRAG(matrix_a, experimental::precision::b1, row_major)
__CPUSIM_SUBFRAG(matrix_b, experimental::precision::b1, col_major)
#undef __CPUSIM_SUBFRAG
template <typename Use, typename T, typename L>
__device__ __forceinline__ void load_matrix_sync(fragment<Use, 8, 8, __cpusim::subbyte<T>::k, T, L>& f, const void* p,
                                                 unsigned ldm) {
    const unsigned l = __cpusim_laneid();
    f.x[0] = static_cast<const unsigned*>(p)[(l / 4) * (ldm * __cpusim::subbyte<T>::bits / 32) + l % 4];
}
template <typename T>
__device__ __forceinline__ void fill_fragment(fragment<matrix_a, 8, 8, __cpusim::subbyte<T>::k, T, row_major>& f,
                                              unsigned v) {
    f.x[0] = v;
}
template <typename T>
__device__ __forceinline__ void fill_fragment(fragment<matrix_b, 8, 8, __cpusim::subbyte<T>::k, T, col_major>& f,
                                              unsigned v) {
    f.x[0] = v;
}
__device__ __forceinline__ void mma_sync(fragment<accumulator, 8, 8, 32, int>& d,
                                         const fragment<matrix_a, 8, 8, 32, experimental::precision::s4, row_major>& a,
                                         const fragment<matrix_b, 8, 8, 32, experimental::precision::s4, col_major>& b,
                                         const fragment<accumulator, 8, 8, 32, int>& c, bool satf = false) {
    const int r0 = __cpusim_mmav_s4_s32(0, a.x, b.x[0], c.x[0], c.x[1]);
    const int r1 = __cpusim_mmav_s4_s32(1, a.x, b.x[0], c.x[0], c.x[1]);
    d.x[0] = r0, d.x[1] = r1;
}
__device__ __forceinline__ void mma_sync(fragment<accumulator, 8, 8, 32, int>& d,
                                         const fragment<matrix_a, 8, 8, 32, experimental::precision::u4, row_major>& a,
                                         const fragment<matrix_b, 8, 8, 32, experimental::precision::u4, col_major>& b,
                                         const fragment<accumulator, 8, 8, 32, int>& c, bool satf = false) {
    const int r0 = __cpusim_mmav_u4_s32(0, a.x, b.x[0], c.x[0], c.x[1]);
    const int r1 = __cpusim_mmav_u4_s32(1, a.x, b.x[0], c.x[0], c.x[1]);
    d.x[0] = r0, d.x[1] = r1;
}
// D = C + popc(A op B) per row of A and column of B.
__device__ __forceinline__ void bmma_sync(fragment<accumulator, 8, 8, 128, int>& d,
                                          const fragment<matrix_a, 8, 8, 128, experimental::precision::b1, row_major>& a,
                                          const fragment<matrix_b, 8, 8, 128, experimental::precision::b1, col_major>& b,
                                          const fragment<accumulator, 8, 8, 128, int>& c,
                                          experimental::bmmaBitOp op = experimental::bmmaBitOpXOR,
                                          experimental::bmmaAccumulateOp = experimental::bmmaAccumulateOpPOPC) {
    int r0, r1;
    if (op == experimental::bmmaBitOpAND) {
        r0 = __cpusim_mmav_b1and_s32(0, a.x, b.x[0], c.x[0], c.x[1]);
        r1 = __cpusim_mmav_b1and_s32(1, a.x, b.x[0], c.x[0], c.x[1]);
    } else {
        r0 = __cpusim_mmav_b1xor_s32(0, a.x, b.x[0], c.x[0], c.x[1]);
        r1 = __cpusim_mmav_b1xor_s32(1, a.x, b.x[0], c.x[0], c.x[1]);
    }
    d.x[0] = r0, d.x[1] = r1;
}

#undef __CPUSIM_MMA
#undef __CPUSIM_E4
#undef __CPUSIM_E8
#undef __CPUSIM_E16

}  // namespace wmma
}  // namespace nvcuda

#endif
