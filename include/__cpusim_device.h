/* Device-side declarations for CUDA source compiled by the CPU SIMT compiler:
 * execution space attributes, built-in variables, synchronization and the
 * hooks Clang needs to lower <<<...>>>. Included by cuda_runtime.h when the
 * file is compiled as CUDA (__CUDA__). */
#ifndef CPUSIM_DEVICE_H
#define CPUSIM_DEVICE_H

#include <math.h> /* the host math functions, next to the device ones below (as nvcc) */
#include <stdlib.h>

#include "cuda_runtime_api.h"

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __noinline__ __attribute__((noinline))
#define __align__(n) __attribute__((aligned(n)))
#define __restrict__ __restrict
#define __grid_constant__

/* <<<grid, block, shmem, stream>>> is lowered by Clang into these two calls
 * followed by cudaLaunchKernel from the kernel's host stub. */
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0,
                                                void* stream = 0);
extern "C" cudaError_t __cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem, void* stream);

/* Built-in variables. Each field reads a special register; the SIMT
 * compiler turns these reads into per-lane vectors or warp-uniform scalars. */
#define __CPUSIM_BUILTIN_FIELD(FIELD, READ)                                               \
    __declspec(property(get = __get_##FIELD)) unsigned int FIELD;                          \
    static inline __attribute__((always_inline)) __device__ unsigned int __get_##FIELD(void) { \
        return READ;                                                                       \
    }

#define __CPUSIM_BUILTIN_VAR(TYPE, PREFIX)                          \
    struct TYPE {                                                   \
        __CPUSIM_BUILTIN_FIELD(x, __nvvm_read_ptx_sreg_##PREFIX##_x()) \
        __CPUSIM_BUILTIN_FIELD(y, __nvvm_read_ptx_sreg_##PREFIX##_y()) \
        __CPUSIM_BUILTIN_FIELD(z, __nvvm_read_ptx_sreg_##PREFIX##_z()) \
        __device__ operator uint3() const { return uint3{x, y, z}; } \
        __device__ operator dim3() const { return dim3(x, y, z); }   \
                                                                    \
    private:                                                        \
        __device__ TYPE() = delete;                                 \
        __device__ TYPE(const TYPE&) = delete;                      \
        __device__ void operator=(const TYPE&) const = delete;      \
        __device__ TYPE* operator&() const = delete;                \
    };

__CPUSIM_BUILTIN_VAR(__cpusim_threadIdx_t, tid)
__CPUSIM_BUILTIN_VAR(__cpusim_blockIdx_t, ctaid)
__CPUSIM_BUILTIN_VAR(__cpusim_blockDim_t, ntid)
__CPUSIM_BUILTIN_VAR(__cpusim_gridDim_t, nctaid)

#undef __CPUSIM_BUILTIN_VAR
#undef __CPUSIM_BUILTIN_FIELD

extern const __device__ __cpusim_threadIdx_t threadIdx;
extern const __device__ __cpusim_blockIdx_t blockIdx;
extern const __device__ __cpusim_blockDim_t blockDim;
extern const __device__ __cpusim_gridDim_t gridDim;
static constexpr int warpSize = 32;

/* min/max: CUDA declares integer and floating point overloads usable on both sides. */
#define __CPUSIM_MINMAX(T)                                                                 \
    static inline __host__ __device__ T min(T a, T b) { return a < b ? a : b; }              \
    static inline __host__ __device__ T max(T a, T b) { return a > b ? a : b; }
__CPUSIM_MINMAX(int)
__CPUSIM_MINMAX(unsigned int)
__CPUSIM_MINMAX(long)
__CPUSIM_MINMAX(unsigned long)
__CPUSIM_MINMAX(long long)
__CPUSIM_MINMAX(unsigned long long)
#undef __CPUSIM_MINMAX
static inline __host__ __device__ float min(float a, float b) { return __builtin_fminf(a, b); }
static inline __host__ __device__ float max(float a, float b) { return __builtin_fmaxf(a, b); }
static inline __host__ __device__ double min(double a, double b) { return __builtin_fmin(a, b); }
static inline __host__ __device__ double max(double a, double b) { return __builtin_fmax(a, b); }

/* Math library, device side. Each function maps to a Clang builtin: the
 * ones with an x86 instruction (sqrt, fabs, floor, fma...) become LLVM
 * intrinsics the SIMT compiler widens directly; the transcendental ones (exp,
 * log, pow, sin, erf...) become calls the compiler turns into the AVX-512
 * functions of glibc's libmvec, 16 floats (or 8 doubles) per call. They sit
 * next to the host declarations from <math.h> as __device__ overloads, as in
 * Clang's own CUDA headers. */
#define __CPUSIM_DEV static __device__ __forceinline__
#define __CPUSIM_MATH1(NAME)                                                   \
    __CPUSIM_DEV float NAME##f(float x) { return __builtin_##NAME##f(x); }     \
    __CPUSIM_DEV double NAME(double x) { return __builtin_##NAME(x); }         \
    __CPUSIM_DEV float NAME(float x) { return __builtin_##NAME##f(x); }
#define __CPUSIM_MATH2(NAME)                                                                   \
    __CPUSIM_DEV float NAME##f(float x, float y) { return __builtin_##NAME##f(x, y); }         \
    __CPUSIM_DEV double NAME(double x, double y) { return __builtin_##NAME(x, y); }            \
    __CPUSIM_DEV float NAME(float x, float y) { return __builtin_##NAME##f(x, y); }
__CPUSIM_MATH1(sqrt) __CPUSIM_MATH1(cbrt) __CPUSIM_MATH1(fabs) __CPUSIM_MATH1(floor) __CPUSIM_MATH1(ceil)
__CPUSIM_MATH1(trunc) __CPUSIM_MATH1(round) __CPUSIM_MATH1(rint) __CPUSIM_MATH1(nearbyint)
__CPUSIM_MATH1(exp) __CPUSIM_MATH1(exp2) __CPUSIM_MATH1(exp10) __CPUSIM_MATH1(expm1)
__CPUSIM_MATH1(log) __CPUSIM_MATH1(log2) __CPUSIM_MATH1(log10) __CPUSIM_MATH1(log1p)
__CPUSIM_MATH1(sin) __CPUSIM_MATH1(cos) __CPUSIM_MATH1(tan) __CPUSIM_MATH1(asin) __CPUSIM_MATH1(acos)
__CPUSIM_MATH1(atan) __CPUSIM_MATH1(sinh) __CPUSIM_MATH1(cosh) __CPUSIM_MATH1(tanh) __CPUSIM_MATH1(asinh)
__CPUSIM_MATH1(acosh) __CPUSIM_MATH1(atanh) __CPUSIM_MATH1(erf) __CPUSIM_MATH1(erfc)
__CPUSIM_MATH2(pow) __CPUSIM_MATH2(atan2) __CPUSIM_MATH2(hypot) __CPUSIM_MATH2(fmin) __CPUSIM_MATH2(fmax)
__CPUSIM_MATH2(copysign) __CPUSIM_MATH2(fmod) __CPUSIM_MATH2(fdim)
#undef __CPUSIM_MATH1
#undef __CPUSIM_MATH2
__CPUSIM_DEV float fmaf(float x, float y, float z) { return __builtin_fmaf(x, y, z); }
__CPUSIM_DEV double fma(double x, double y, double z) { return __builtin_fma(x, y, z); }
__CPUSIM_DEV float fma(float x, float y, float z) { return __builtin_fmaf(x, y, z); }
__CPUSIM_DEV float abs(float x) { return __builtin_fabsf(x); }
__CPUSIM_DEV double abs(double x) { return __builtin_fabs(x); }
__CPUSIM_DEV float powf(float x, int n) { return __builtin_powf(x, (float)n); }
__CPUSIM_DEV double pow(double x, int n) { return __builtin_pow(x, (double)n); }
__CPUSIM_DEV float rsqrtf(float x) { return 1.0f / __builtin_sqrtf(x); }
__CPUSIM_DEV double rsqrt(double x) { return 1.0 / __builtin_sqrt(x); }
__CPUSIM_DEV float rcbrtf(float x) { return 1.0f / __builtin_cbrtf(x); }
__CPUSIM_DEV void sincosf(float x, float* s, float* c) { *s = __builtin_sinf(x); *c = __builtin_cosf(x); }
__CPUSIM_DEV void sincos(double x, double* s, double* c) { *s = __builtin_sin(x); *c = __builtin_cos(x); }
__CPUSIM_DEV float sinpif(float x) { return __builtin_sinf(3.14159265358979f * x); }
__CPUSIM_DEV float cospif(float x) { return __builtin_cosf(3.14159265358979f * x); }
__CPUSIM_DEV float fdividef(float x, float y) { return x / y; }
__CPUSIM_DEV int isnan(float x) { return __builtin_isnan(x); }
__CPUSIM_DEV int isnan(double x) { return __builtin_isnan(x); }
__CPUSIM_DEV int isinf(float x) { return __builtin_isinf(x); }
__CPUSIM_DEV int isinf(double x) { return __builtin_isinf(x); }
__CPUSIM_DEV int isfinite(float x) { return __builtin_isfinite(x); }
__CPUSIM_DEV int isfinite(double x) { return __builtin_isfinite(x); }
__CPUSIM_DEV int signbit(float x) { return __builtin_signbit(x); }
__CPUSIM_DEV int signbit(double x) { return __builtin_signbit(x); }
/* The fast intrinsics: same functions here (no lower-precision hardware path). */
__CPUSIM_DEV float __expf(float x) { return __builtin_expf(x); }
__CPUSIM_DEV float __exp10f(float x) { return __builtin_exp10f(x); }
__CPUSIM_DEV float __logf(float x) { return __builtin_logf(x); }
__CPUSIM_DEV float __log2f(float x) { return __builtin_log2f(x); }
__CPUSIM_DEV float __log10f(float x) { return __builtin_log10f(x); }
__CPUSIM_DEV float __powf(float x, float y) { return __builtin_powf(x, y); }
__CPUSIM_DEV float __sinf(float x) { return __builtin_sinf(x); }
__CPUSIM_DEV float __cosf(float x) { return __builtin_cosf(x); }
__CPUSIM_DEV float __tanf(float x) { return __builtin_tanf(x); }
__CPUSIM_DEV void __sincosf(float x, float* s, float* c) { *s = __builtin_sinf(x); *c = __builtin_cosf(x); }
__CPUSIM_DEV float __fdividef(float x, float y) { return x / y; }
__CPUSIM_DEV float __frcp_rn(float x) { return 1.0f / x; }
__CPUSIM_DEV float __fsqrt_rn(float x) { return __builtin_sqrtf(x); }
__CPUSIM_DEV float __frsqrt_rn(float x) { return 1.0f / __builtin_sqrtf(x); }
__CPUSIM_DEV float __saturatef(float x) { return __builtin_fminf(__builtin_fmaxf(x, 0.0f), 1.0f); }
__CPUSIM_DEV float __fadd_rn(float x, float y) { return x + y; }
__CPUSIM_DEV float __fsub_rn(float x, float y) { return x - y; }
__CPUSIM_DEV float __fmul_rn(float x, float y) { return x * y; }
__CPUSIM_DEV float __fdiv_rn(float x, float y) { return x / y; }
__CPUSIM_DEV float __fmaf_rn(float x, float y, float z) { return __builtin_fmaf(x, y, z); }
__CPUSIM_DEV double __dadd_rn(double x, double y) { return x + y; }
__CPUSIM_DEV double __dmul_rn(double x, double y) { return x * y; }
__CPUSIM_DEV double __fma_rn(double x, double y, double z) { return __builtin_fma(x, y, z); }

/* Warp-level operations. A warp's 32 threads are 32 consecutive lanes of the
 * same vectors and run in lockstep, so everything here sits on two
 * primitives the SIMT compiler lowers:
 *   __cpusim_shfl(v, src): v of lane `src` (0..31) of the caller's warp
 *                           (a permute inside each 32-lane segment);
 *   __cpusim_ballot(p):     bit i set if lane i of the warp is active here
 *                           and p != 0 (the active mask at this point).
 * __syncwarp is a no-op: the lanes of a warp never drift apart. */
extern "C" __device__ __attribute__((convergent)) int __cpusim_shfl(int v, int src_lane);
extern "C" __device__ __attribute__((convergent)) unsigned __cpusim_ballot(int pred);

__CPUSIM_DEV unsigned __cpusim_laneid() { return __nvvm_read_ptx_sreg_laneid(); }

/* Any trivially copyable type, 4 bytes at a time. */
template <typename T>
__CPUSIM_DEV T __cpusim_shfl_any(T v, int src) {
    constexpr int words = (sizeof(T) + 3) / 4;
    int w[words] = {};
    __builtin_memcpy(w, &v, sizeof(T));
#pragma unroll
    for (int i = 0; i < words; ++i) w[i] = __cpusim_shfl(w[i], src);
    T r;
    __builtin_memcpy(&r, w, sizeof(T));
    return r;
}

template <typename T>
__CPUSIM_DEV T __shfl_sync(unsigned mask, T var, int srcLane, int width = warpSize) {
    int lane = (int)__cpusim_laneid(), base = lane & ~(width - 1);
    return __cpusim_shfl_any(var, base + (srcLane & (width - 1)));
}
template <typename T>
__CPUSIM_DEV T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width = warpSize) {
    int lane = (int)__cpusim_laneid(), base = lane & ~(width - 1), src = lane - (int)delta;
    return __cpusim_shfl_any(var, src < base ? lane : src);
}
template <typename T>
__CPUSIM_DEV T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width = warpSize) {
    int lane = (int)__cpusim_laneid(), base = lane & ~(width - 1), src = lane + (int)delta;
    return __cpusim_shfl_any(var, src >= base + width ? lane : src);
}
template <typename T>
__CPUSIM_DEV T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = warpSize) {
    int lane = (int)__cpusim_laneid(), base = lane & ~(width - 1), src = lane ^ laneMask;
    return __cpusim_shfl_any(var, src >= base + width ? lane : src);
}
/* The pre-Volta forms (whole warp). */
template <typename T>
__CPUSIM_DEV T __shfl(T var, int srcLane, int width = warpSize) { return __shfl_sync(0xffffffffu, var, srcLane, width); }
template <typename T>
__CPUSIM_DEV T __shfl_up(T var, unsigned int delta, int width = warpSize) { return __shfl_up_sync(0xffffffffu, var, delta, width); }
template <typename T>
__CPUSIM_DEV T __shfl_down(T var, unsigned int delta, int width = warpSize) {
    return __shfl_down_sync(0xffffffffu, var, delta, width);
}
template <typename T>
__CPUSIM_DEV T __shfl_xor(T var, int laneMask, int width = warpSize) { return __shfl_xor_sync(0xffffffffu, var, laneMask, width); }

__CPUSIM_DEV unsigned __activemask() { return __cpusim_ballot(1); }
__CPUSIM_DEV unsigned __ballot_sync(unsigned mask, int pred) { return __cpusim_ballot(pred) & mask; }
__CPUSIM_DEV int __any_sync(unsigned mask, int pred) { return __ballot_sync(mask, pred) != 0; }
__CPUSIM_DEV int __all_sync(unsigned mask, int pred) { return __ballot_sync(mask, !pred) == 0; }
__CPUSIM_DEV int __uni_sync(unsigned mask, int pred) {
    unsigned b = __ballot_sync(mask, pred);
    return b == 0 || b == (__activemask() & mask);
}
__CPUSIM_DEV unsigned __ballot(int pred) { return __cpusim_ballot(pred); }
__CPUSIM_DEV int __any(int pred) { return __cpusim_ballot(pred) != 0; }
__CPUSIM_DEV int __all(int pred) { return __cpusim_ballot(!pred) == 0; }
__CPUSIM_DEV void __syncwarp(unsigned mask = 0xffffffffu) {}

/* Lanes of the warp holding the same value. */
template <typename T>
__CPUSIM_DEV unsigned __match_any_sync(unsigned mask, T value) {
    unsigned r = 0;
#pragma unroll
    for (int i = 0; i < 32; ++i) r |= (unsigned)(__shfl_sync(mask, value, i) == value) << i;
    return r & __activemask() & mask;
}
template <typename T>
__CPUSIM_DEV unsigned __match_all_sync(unsigned mask, T value, int* pred) {
    unsigned want = __activemask() & mask;
    *pred = __match_any_sync(mask, value) == want;
    return *pred ? want : 0;
}

/* Warp reductions (sm_80): every lane of `mask` gets the result. */
#define __CPUSIM_REDUCE(NAME, T, OP)                                    \
    __CPUSIM_DEV T __reduce_##NAME##_sync(unsigned mask, T v) {         \
        for (int o = 16; o > 0; o >>= 1) {                              \
            T w = __shfl_xor_sync(mask, v, o);                          \
            v = OP;                                                     \
        }                                                               \
        return v;                                                       \
    }
__CPUSIM_REDUCE(add, unsigned, v + w) __CPUSIM_REDUCE(add, int, v + w)
__CPUSIM_REDUCE(min, unsigned, v < w ? v : w) __CPUSIM_REDUCE(min, int, v < w ? v : w)
__CPUSIM_REDUCE(max, unsigned, v > w ? v : w) __CPUSIM_REDUCE(max, int, v > w ? v : w)
__CPUSIM_REDUCE(and, unsigned, v & w) __CPUSIM_REDUCE(or, unsigned, v | w) __CPUSIM_REDUCE(xor, unsigned, v ^ w)
#undef __CPUSIM_REDUCE

/* Bit reinterpretation and conversions. */
__CPUSIM_DEV int __float_as_int(float x) { return __builtin_bit_cast(int, x); }
__CPUSIM_DEV unsigned __float_as_uint(float x) { return __builtin_bit_cast(unsigned, x); }
__CPUSIM_DEV float __int_as_float(int x) { return __builtin_bit_cast(float, x); }
__CPUSIM_DEV float __uint_as_float(unsigned x) { return __builtin_bit_cast(float, x); }
__CPUSIM_DEV long long __double_as_longlong(double x) { return __builtin_bit_cast(long long, x); }
__CPUSIM_DEV double __longlong_as_double(long long x) { return __builtin_bit_cast(double, x); }
__CPUSIM_DEV int __double2hiint(double x) { return (int)(__builtin_bit_cast(unsigned long long, x) >> 32); }
__CPUSIM_DEV int __double2loint(double x) { return (int)__builtin_bit_cast(unsigned long long, x); }
__CPUSIM_DEV double __hiloint2double(int hi, int lo) {
    return __builtin_bit_cast(double, ((unsigned long long)(unsigned)hi << 32) | (unsigned)lo);
}
__CPUSIM_DEV int __float2int_rn(float x) { return (int)__builtin_rintf(x); }
__CPUSIM_DEV int __float2int_rz(float x) { return (int)x; }
__CPUSIM_DEV int __float2int_rd(float x) { return (int)__builtin_floorf(x); }
__CPUSIM_DEV int __float2int_ru(float x) { return (int)__builtin_ceilf(x); }
__CPUSIM_DEV unsigned __float2uint_rn(float x) { return (unsigned)__builtin_rintf(x); }
__CPUSIM_DEV unsigned __float2uint_rz(float x) { return (unsigned)x; }
__CPUSIM_DEV long long __float2ll_rn(float x) { return (long long)__builtin_rintf(x); }
__CPUSIM_DEV int __double2int_rn(double x) { return (int)__builtin_rint(x); }
__CPUSIM_DEV int __double2int_rz(double x) { return (int)x; }
__CPUSIM_DEV float __double2float_rn(double x) { return (float)x; }
__CPUSIM_DEV float __int2float_rn(int x) { return (float)x; }
__CPUSIM_DEV float __uint2float_rn(unsigned x) { return (float)x; }
__CPUSIM_DEV double __int2double_rn(int x) { return (double)x; }
__CPUSIM_DEV float __ll2float_rn(long long x) { return (float)x; }

/* Integer intrinsics. */
__CPUSIM_DEV int __popc(unsigned x) { return __builtin_popcount(x); }
__CPUSIM_DEV int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
__CPUSIM_DEV int __clz(int x) { return x ? __builtin_clz((unsigned)x) : 32; }
__CPUSIM_DEV int __clzll(long long x) { return x ? __builtin_clzll((unsigned long long)x) : 64; }
__CPUSIM_DEV int __ffs(int x) { return __builtin_ffs(x); }
__CPUSIM_DEV int __ffsll(long long x) { return __builtin_ffsll(x); }
__CPUSIM_DEV unsigned __brev(unsigned x) { return __builtin_bitreverse32(x); }
__CPUSIM_DEV unsigned long long __brevll(unsigned long long x) { return __builtin_bitreverse64(x); }
__CPUSIM_DEV int __mulhi(int x, int y) { return (int)(((long long)x * y) >> 32); }
__CPUSIM_DEV unsigned __umulhi(unsigned x, unsigned y) { return (unsigned)(((unsigned long long)x * y) >> 32); }
__CPUSIM_DEV long long __mul64hi(long long x, long long y) { return (long long)(((__int128)x * y) >> 64); }
__CPUSIM_DEV unsigned long long __umul64hi(unsigned long long x, unsigned long long y) {
    return (unsigned long long)(((unsigned __int128)x * y) >> 64);
}
__CPUSIM_DEV int __mul24(int x, int y) { return (x << 8 >> 8) * (y << 8 >> 8); }
__CPUSIM_DEV unsigned __umul24(unsigned x, unsigned y) { return (x & 0xffffff) * (y & 0xffffff); }
__CPUSIM_DEV int __hadd(int x, int y) { return (int)(((long long)x + y) >> 1); }
__CPUSIM_DEV unsigned __uhadd(unsigned x, unsigned y) { return (unsigned)(((unsigned long long)x + y) >> 1); }
__CPUSIM_DEV unsigned __sad(int x, int y, unsigned z) { return z + (unsigned)(x > y ? x - y : y - x); }
__CPUSIM_DEV unsigned __usad(unsigned x, unsigned y, unsigned z) { return z + (x > y ? x - y : y - x); }
__CPUSIM_DEV unsigned __byte_perm(unsigned x, unsigned y, unsigned s) {
    unsigned long long xy = ((unsigned long long)y << 32) | x;
    unsigned r = 0;
    for (int i = 0; i < 4; ++i) r |= (unsigned)((xy >> (8 * ((s >> (4 * i)) & 7))) & 0xff) << (8 * i);
    return r;
}
__CPUSIM_DEV unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned s) {
    return (unsigned)((((unsigned long long)hi << 32 | lo) << (s & 31)) >> 32);
}
__CPUSIM_DEV unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned s) {
    return (unsigned)(((unsigned long long)hi << 32 | lo) >> (s & 31));
}

/* 8 and 16-bit dot products in a 32-bit word (sm_61). */
__CPUSIM_DEV int __dp4a(int a, int b, int c) {
    for (int i = 0; i < 4; ++i) c += (int)(signed char)(a >> (8 * i)) * (int)(signed char)(b >> (8 * i));
    return c;
}
__CPUSIM_DEV unsigned __dp4a(unsigned a, unsigned b, unsigned c) {
    for (int i = 0; i < 4; ++i) c += ((a >> (8 * i)) & 0xffu) * ((b >> (8 * i)) & 0xffu);
    return c;
}
__CPUSIM_DEV int __dp4a(char4 a, char4 b, int c) { return __dp4a(__builtin_bit_cast(int, a), __builtin_bit_cast(int, b), c); }
__CPUSIM_DEV unsigned __dp4a(uchar4 a, uchar4 b, unsigned c) {
    return __dp4a(__builtin_bit_cast(unsigned, a), __builtin_bit_cast(unsigned, b), c);
}
// Two 16-bit values of a times two bytes of b: the low or the high pair of b.
__CPUSIM_DEV int __dp2a_lo(int a, int b, int c) {
    return c + (int)(short)a * (int)(signed char)b + (int)(short)(a >> 16) * (int)(signed char)(b >> 8);
}
__CPUSIM_DEV int __dp2a_hi(int a, int b, int c) {
    return c + (int)(short)a * (int)(signed char)(b >> 16) + (int)(short)(a >> 16) * (int)(signed char)(b >> 24);
}
__CPUSIM_DEV unsigned __dp2a_lo(unsigned a, unsigned b, unsigned c) {
    return c + (a & 0xffffu) * (b & 0xffu) + (a >> 16) * ((b >> 8) & 0xffu);
}
__CPUSIM_DEV unsigned __dp2a_hi(unsigned a, unsigned b, unsigned c) {
    return c + (a & 0xffffu) * ((b >> 16) & 0xffu) + (a >> 16) * (b >> 24);
}

/* SIMD within a word: 2 x 16 or 4 x 8 bits, each lane wrapping. */
#define __CPUSIM_SIMD(NAME, BITS, T, EXPR)                                                     \
    __CPUSIM_DEV unsigned NAME(unsigned a, unsigned b) {                                        \
        unsigned r = 0;                                                                         \
        for (int i = 0; i < 32; i += BITS) {                                                    \
            T x = (T)(a >> i), y = (T)(b >> i);                                                 \
            r |= ((unsigned)(EXPR) & ((1u << BITS) - 1)) << i;                                  \
        }                                                                                       \
        return r;                                                                               \
    }
__CPUSIM_SIMD(__vadd2, 16, unsigned short, x + y)
__CPUSIM_SIMD(__vsub2, 16, unsigned short, x - y)
__CPUSIM_SIMD(__vadd4, 8, unsigned char, x + y)
__CPUSIM_SIMD(__vsub4, 8, unsigned char, x - y)
__CPUSIM_SIMD(__vabsdiffu2, 16, unsigned short, x > y ? x - y : y - x)
__CPUSIM_SIMD(__vabsdiffu4, 8, unsigned char, x > y ? x - y : y - x)
__CPUSIM_SIMD(__vabsdiffs2, 16, short, x > y ? x - y : y - x)
__CPUSIM_SIMD(__vabsdiffs4, 8, signed char, x > y ? x - y : y - x)
__CPUSIM_SIMD(__vmaxu2, 16, unsigned short, x > y ? x : y)
__CPUSIM_SIMD(__vmaxu4, 8, unsigned char, x > y ? x : y)
__CPUSIM_SIMD(__vmaxs2, 16, short, x > y ? x : y)
__CPUSIM_SIMD(__vmaxs4, 8, signed char, x > y ? x : y)
__CPUSIM_SIMD(__vminu2, 16, unsigned short, x < y ? x : y)
__CPUSIM_SIMD(__vminu4, 8, unsigned char, x < y ? x : y)
__CPUSIM_SIMD(__vmins2, 16, short, x < y ? x : y)
__CPUSIM_SIMD(__vmins4, 8, signed char, x < y ? x : y)
__CPUSIM_SIMD(__vavgu2, 16, unsigned short, ((unsigned)x + y + 1) >> 1)
__CPUSIM_SIMD(__vavgu4, 8, unsigned char, ((unsigned)x + y + 1) >> 1)
__CPUSIM_SIMD(__vcmpeq2, 16, unsigned short, x == y ? 0xffffu : 0u)
__CPUSIM_SIMD(__vcmpeq4, 8, unsigned char, x == y ? 0xffu : 0u)
__CPUSIM_SIMD(__vcmpne2, 16, unsigned short, x != y ? 0xffffu : 0u)
__CPUSIM_SIMD(__vcmpne4, 8, unsigned char, x != y ? 0xffu : 0u)
__CPUSIM_SIMD(__vcmpgtu4, 8, unsigned char, x > y ? 0xffu : 0u)
__CPUSIM_SIMD(__vcmpgts4, 8, signed char, x > y ? 0xffu : 0u)
#undef __CPUSIM_SIMD
__CPUSIM_DEV unsigned __vsadu4(unsigned a, unsigned b) {
    unsigned d = __vabsdiffu4(a, b);
    return (d & 0xffu) + ((d >> 8) & 0xffu) + ((d >> 16) & 0xffu) + (d >> 24);
}

/* Cached loads and stores: the cache hints mean nothing on the CPU. */
template <typename T>
__CPUSIM_DEV T __ldg(const T* p) { return *p; }
template <typename T>
__CPUSIM_DEV T __ldca(const T* p) { return *p; }
template <typename T>
__CPUSIM_DEV T __ldcg(const T* p) { return *p; }
template <typename T>
__CPUSIM_DEV T __ldcs(const T* p) { return *p; }
template <typename T>
__CPUSIM_DEV T __ldlu(const T* p) { return *p; }
template <typename T>
__CPUSIM_DEV T __ldcv(const T* p) { return *p; }
template <typename T>
__CPUSIM_DEV void __stwb(T* p, T v) { *p = v; }
template <typename T>
__CPUSIM_DEV void __stcg(T* p, T v) { *p = v; }
// Streaming store (evict-first, data not reused soon): a non-temporal store,
// which skips reading the line before writing it. The compiler emits it where
// the whole warp writes contiguous lanes; elsewhere it is an ordinary store.
template <typename T>
__CPUSIM_DEV void __stcs(T* p, T v) {
    if constexpr (__is_arithmetic(T)) __builtin_nontemporal_store(v, p);
    else *p = v;
}
template <typename T>
__CPUSIM_DEV void __stwt(T* p, T v) { *p = v; }
#undef __CPUSIM_DEV

#endif
