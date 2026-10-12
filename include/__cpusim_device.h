/* Device-side declarations for CUDA source compiled by the CPU SIMT compiler:
 * execution space attributes, built-in variables, synchronization and the
 * hooks Clang needs to lower <<<...>>>. Included by cuda_runtime.h when the
 * file is compiled as CUDA (__CUDA__). */
#ifndef CPUSIM_DEVICE_H
#define CPUSIM_DEVICE_H

#include <math.h> /* the host math functions, next to the device ones below (as nvcc) */
#include <limits.h>
#include <stdlib.h>

#include "cuda_runtime_api.h"

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
/* Clang keeps __managed__ for HIP: here a __device__ variable the compiler
 * marks (its section), which the runtime keeps equal to its host copy at the
 * kernels' boundaries (copied in before each launch, back after it). */
#define __managed__ __device__ __attribute__((section("cpusim_managed")))
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
/* The transcendental functions in single precision: computed in double and
 * rounded once. libmvec's float versions err by up to ~4 ulp, above the
 * bounds CUDA documents (1 to 2 ulp for most); its double versions stay under
 * 1 ulp, and one rounding to float adds at most half of one. The fast
 * intrinsics (__expf, __logf...) keep the float versions, as CUDA gives them
 * looser bounds. */
#define __CPUSIM_MATH1D(NAME)                                                              \
    __CPUSIM_DEV float NAME##f(float x) { return (float)__builtin_##NAME((double)x); }     \
    __CPUSIM_DEV double NAME(double x) { return __builtin_##NAME(x); }                     \
    __CPUSIM_DEV float NAME(float x) { return (float)__builtin_##NAME((double)x); }
#define __CPUSIM_MATH2D(NAME)                                                                                   \
    __CPUSIM_DEV float NAME##f(float x, float y) { return (float)__builtin_##NAME((double)x, (double)y); }     \
    __CPUSIM_DEV double NAME(double x, double y) { return __builtin_##NAME(x, y); }                            \
    __CPUSIM_DEV float NAME(float x, float y) { return (float)__builtin_##NAME((double)x, (double)y); }
__CPUSIM_MATH1(sqrt) __CPUSIM_MATH1(fabs) __CPUSIM_MATH1(floor) __CPUSIM_MATH1(ceil)
__CPUSIM_MATH1(trunc) __CPUSIM_MATH1(round) __CPUSIM_MATH1(rint) __CPUSIM_MATH1(nearbyint)
__CPUSIM_MATH1D(exp) __CPUSIM_MATH1D(exp2) __CPUSIM_MATH1D(exp10) __CPUSIM_MATH1D(expm1)
__CPUSIM_MATH1D(log) __CPUSIM_MATH1D(log2) __CPUSIM_MATH1D(log10) __CPUSIM_MATH1D(log1p)
__CPUSIM_MATH1D(sin) __CPUSIM_MATH1D(cos) __CPUSIM_MATH1D(tan) __CPUSIM_MATH1D(asin) __CPUSIM_MATH1D(acos)
__CPUSIM_MATH1D(atan) __CPUSIM_MATH1D(sinh) __CPUSIM_MATH1D(cosh) __CPUSIM_MATH1D(tanh) __CPUSIM_MATH1D(asinh)
__CPUSIM_MATH1D(acosh) __CPUSIM_MATH1D(atanh) __CPUSIM_MATH1D(erf) __CPUSIM_MATH1D(erfc)
__CPUSIM_MATH2D(pow) __CPUSIM_MATH2D(atan2) __CPUSIM_MATH2(hypot) __CPUSIM_MATH2(fmin) __CPUSIM_MATH2(fmax)
__CPUSIM_MATH2(copysign) __CPUSIM_MATH2(fmod) __CPUSIM_MATH2(fdim)
#undef __CPUSIM_MATH1
#undef __CPUSIM_MATH2
#undef __CPUSIM_MATH1D
#undef __CPUSIM_MATH2D
__CPUSIM_DEV float fmaf(float x, float y, float z) { return __builtin_fmaf(x, y, z); }
__CPUSIM_DEV double fma(double x, double y, double z) { return __builtin_fma(x, y, z); }
__CPUSIM_DEV float fma(float x, float y, float z) { return __builtin_fmaf(x, y, z); }
__CPUSIM_DEV int abs(int x) { return x < 0 ? -x : x; }
__CPUSIM_DEV long abs(long x) { return x < 0 ? -x : x; }
__CPUSIM_DEV long long abs(long long x) { return x < 0 ? -x : x; }
__CPUSIM_DEV float abs(float x) { return __builtin_fabsf(x); }
__CPUSIM_DEV double abs(double x) { return __builtin_fabs(x); }
__CPUSIM_DEV float powf(float x, int n) { return (float)__builtin_pow((double)x, (double)n); }
__CPUSIM_DEV double pow(double x, int n) { return __builtin_pow(x, (double)n); }
__CPUSIM_DEV float rsqrtf(float x) { return 1.0f / __builtin_sqrtf(x); }
/* 1/sqrt rounded twice errs by more than an ulp; one Newton step on the
 * residual of y^2 x brings it within one. */
__CPUSIM_DEV double rsqrt(double x) {
    double y = 1.0 / __builtin_sqrt(x);
    if (!(x > 0) || __builtin_isinf(x)) return y;
    double r = __builtin_fma(-x * y, y, 1.0);
    return __builtin_fma(0.5 * y, r, y);
}
/* cbrt: the C library's, then one Newton step with y^3 - x computed as a
 * sum of two doubles (glibc's own errs by up to ~3 ulp). */
__CPUSIM_DEV double cbrt(double x) {
    /* Subnormal and tiny: scaled by 2^999 first (the cube of y must not
     * underflow), the root by 2^-333. */
    if (x != 0 && __builtin_fabs(x) < 0x1p-900) return cbrt(x * 0x1p999) * 0x1p-333;
    if (__builtin_fabs(x) > 0x1p1000 && !__builtin_isinf(x)) return cbrt(x * 0x1p-999) * 0x1p333;
    double y = __builtin_cbrt(x);
    if (y == 0 || __builtin_isinf(y) || y != y) return y;
    double y2 = y * y, e2 = __builtin_fma(y, y, -y2);
    double y3 = y2 * y, e3 = __builtin_fma(y2, y, -y3) + e2 * y;
    return y - ((y3 - x) + e3) / (3.0 * y2);
}
__CPUSIM_DEV float cbrtf(float x) { return (float)__builtin_cbrt((double)x); }
__CPUSIM_DEV float cbrt(float x) { return (float)__builtin_cbrt((double)x); }
__CPUSIM_DEV float rcbrtf(float x) { return (float)(1.0 / __builtin_cbrt((double)x)); }
__CPUSIM_DEV double rcbrt(double x) {
    double y = 1.0 / cbrt(x);  /* Newton on x y^3 = 1 */
    if (y == 0 || __builtin_isinf(y) || y != y) return y;
    double y2 = y * y, e2 = __builtin_fma(y, y, -y2);
    double y3 = y2 * y, e3 = __builtin_fma(y2, y, -y3) + e2 * y;
    double xy3 = x * y3, e4 = __builtin_fma(x, y3, -xy3) + x * e3;
    return y - y * ((xy3 - 1.0) + e4) / 3.0;
}
__CPUSIM_DEV void sincosf(float x, float* s, float* c) {
    *s = (float)__builtin_sin((double)x);
    *c = (float)__builtin_cos((double)x);
}
__CPUSIM_DEV void sincos(double x, double* s, double* c) { *s = __builtin_sin(x); *c = __builtin_cos(x); }
/* sin(pi x), cos(pi x): x reduced exactly to [-1/2, 1/2] (x - 2 round(x / 2)
 * and 1 - r lose nothing), then pi r as a sum of two doubles, so the only
 * error left is that of sin itself. Exact zeros at the integers, as CUDA's. */
__CPUSIM_DEV double __cpusim_sin_pi_reduced(double r) {  /* |r| <= 1/2 */
    const double pi_hi = 3.141592653589793116, pi_lo = 1.2246467991473532e-16;
    double hi = r * pi_hi, lo = __builtin_fma(r, pi_hi, -hi) + r * pi_lo;
    return __builtin_sin(hi) + __builtin_cos(hi) * lo;
}
__CPUSIM_DEV double sinpi(double x) {
    if (__builtin_isinf(x)) return __builtin_nan("");
    double r = x - 2.0 * __builtin_rint(x * 0.5);
    if (r == __builtin_trunc(r)) return __builtin_copysign(0.0, x);
    if (r > 0.5) r = 1.0 - r;
    else if (r < -0.5) r = -1.0 - r;
    return __cpusim_sin_pi_reduced(r);
}
__CPUSIM_DEV double cospi(double x) {
    if (__builtin_isinf(x)) return __builtin_nan("");
    double r = __builtin_fabs(x - 2.0 * __builtin_rint(x * 0.5));  /* [0, 1] */
    if (r == 0.5) return 0.0;
    return __cpusim_sin_pi_reduced(0.5 - r);
}
__CPUSIM_DEV float sinpif(float x) { return (float)sinpi((double)x); }
__CPUSIM_DEV float cospif(float x) { return (float)cospi((double)x); }
__CPUSIM_DEV void sincospif(float x, float* s, float* c) { *s = sinpif(x); *c = cospif(x); }
__CPUSIM_DEV void sincospi(double x, double* s, double* c) { *s = sinpi(x); *c = cospi(x); }
__CPUSIM_DEV float fdividef(float x, float y) { return x / y; }
__CPUSIM_DEV int isnan(float x) { return __builtin_isnan(x); }
__CPUSIM_DEV int isnan(double x) { return __builtin_isnan(x); }
__CPUSIM_DEV int isinf(float x) { return __builtin_isinf(x); }
__CPUSIM_DEV int isinf(double x) { return __builtin_isinf(x); }
__CPUSIM_DEV int isfinite(float x) { return __builtin_isfinite(x); }
__CPUSIM_DEV int isfinite(double x) { return __builtin_isfinite(x); }
__CPUSIM_DEV int signbit(float x) { return __builtin_signbit(x); }
__CPUSIM_DEV int signbit(double x) { return __builtin_signbit(x); }
/* The fast intrinsics: no lower-precision hardware path here, so the same
 * accurate functions (libmvec's float versions exceed even their bounds). */
__CPUSIM_DEV float __expf(float x) { return (float)__builtin_exp((double)x); }
__CPUSIM_DEV float __exp10f(float x) { return (float)__builtin_exp10((double)x); }
__CPUSIM_DEV float __logf(float x) { return (float)__builtin_log((double)x); }
__CPUSIM_DEV float __log2f(float x) { return (float)__builtin_log2((double)x); }
__CPUSIM_DEV float __log10f(float x) { return (float)__builtin_log10((double)x); }
__CPUSIM_DEV float __powf(float x, float y) { return (float)__builtin_pow((double)x, (double)y); }
__CPUSIM_DEV float __sinf(float x) { return (float)__builtin_sin((double)x); }
__CPUSIM_DEV float __cosf(float x) { return (float)__builtin_cos((double)x); }
__CPUSIM_DEV float __tanf(float x) { return (float)__builtin_tan((double)x); }
__CPUSIM_DEV void __sincosf(float x, float* s, float* c) {
    *s = (float)__builtin_sin((double)x);
    *c = (float)__builtin_cos((double)x);
}
__CPUSIM_DEV float __fdividef(float x, float y) { return x / y; }
__CPUSIM_DEV float __frcp_rn(float x) { return 1.0f / x; }
__CPUSIM_DEV float __fsqrt_rn(float x) { return __builtin_sqrtf(x); }
__CPUSIM_DEV float __frsqrt_rn(float x) { return (float)(1.0 / __builtin_sqrt((double)x)); }
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
/* The whole warp: a butterfly. Part of it: the lanes in the mask, each read
 * once (the others hold nothing of the reduction). */
#define __CPUSIM_REDUCE(NAME, T, OP)                                    \
    __CPUSIM_DEV T __reduce_##NAME##_sync(unsigned mask, T v) {         \
        if (mask == 0xffffffffu) {                                      \
            for (int o = 16; o > 0; o >>= 1) {                          \
                T w = __shfl_xor_sync(mask, v, o);                      \
                v = OP;                                                 \
            }                                                           \
            return v;                                                   \
        }                                                               \
        T mine = v;                                                     \
        bool first = true;                                              \
        for (int src = 0; src < 32; ++src) {                            \
            T w = __shfl_sync(mask, mine, src);                         \
            if (!(mask >> src & 1)) continue;                           \
            if (first) v = w, first = false;                           \
            else v = OP;                                                \
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
__CPUSIM_DEV unsigned __sad(int x, int y, unsigned z) { return z + (unsigned)(x > y ? (long long)x - y : (long long)y - x); }
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

/* Rounding toward zero, up or down. The rounded-to-nearest result r and the
 * exact residual (exact result - r: TwoSum for sums, the fma for products,
 * the remainder for quotients and roots) say on which side the exact value
 * lies; one step toward it, or toward zero, gives the directed result. */
#define __CPUSIM_DIRECTED(T, SUF, NEXT, BIG)                                                          \
    __CPUSIM_DEV T __cpusim_dir_##SUF(T r, T residual, int mode) { /* mode: 0 rz, 1 ru, 2 rd */       \
        if (r != r) return r;                                                                       \
        if (__builtin_isinf(r)) { /* overflow: rounded to nearest, infinity */                      \
            if (mode == 0 || (mode == 1 && r < 0) || (mode == 2 && r > 0))                          \
                return __builtin_copysign((T)BIG, r);                                               \
            return r;                                                                               \
        }                                                                                           \
        if (residual == 0) return r;                                                                \
        if (mode == 1) return residual > 0 ? NEXT(r, (T)__builtin_inf()) : r;                       \
        if (mode == 2) return residual < 0 ? NEXT(r, -(T)__builtin_inf()) : r;                      \
        return (residual < 0) != (r < 0) && r != 0 ? NEXT(r, (T)0) : r;                             \
    }                                                                                               \
    __CPUSIM_DEV T __cpusim_add_##SUF(T a, T b, int mode) {                                         \
        T s = a + b, bb = s - a, e = (a - (s - bb)) + (b - bb);                                     \
        return __cpusim_dir_##SUF(s, e, mode);                                                      \
    }                                                                                               \
    __CPUSIM_DEV T __cpusim_mul_##SUF(T a, T b, int mode) {                                         \
        T p = a * b;                                                                                \
        return __cpusim_dir_##SUF(p, __builtin_fma##SUF(a, b, -p), mode);                           \
    }                                                                                               \
    __CPUSIM_DEV T __cpusim_div_##SUF(T a, T b, int mode) {                                         \
        T q = a / b, rem = __builtin_fma##SUF(-q, b, a);                                            \
        return __cpusim_dir_##SUF(q, b < 0 ? -rem : rem, mode);                                     \
    }                                                                                               \
    __CPUSIM_DEV T __cpusim_sqrt_##SUF(T a, int mode) {                                             \
        T r = __builtin_sqrt##SUF(a);                                                               \
        return __cpusim_dir_##SUF(r, __builtin_fma##SUF(-r, r, a), mode);                           \
    }                                                                                               \
    __CPUSIM_DEV T __cpusim_fma_##SUF(T a, T b, T c, int mode) {                                    \
        T r = __builtin_fma##SUF(a, b, c), p = a * b, pe = __builtin_fma##SUF(a, b, -p);            \
        return __cpusim_dir_##SUF(r, ((p - r) + c) + pe, mode);                                     \
    }
__CPUSIM_DIRECTED(float, f, __builtin_nextafterf, 3.40282347e+38f)
__CPUSIM_DIRECTED(double, , __builtin_nextafter, 1.7976931348623157e+308)
#undef __CPUSIM_DIRECTED
/* Single precision: the operation in double, where the product of two floats
 * is exact and the residual of the rounding to float is too (no subnormal
 * float is subnormal in double). An infinite result there is exact. */
/* Only the residual's sign matters: as a float it could underflow to zero. */
__CPUSIM_DEV float __cpusim_sign_f(double residual) { return residual > 0 ? 1.0f : residual < 0 ? -1.0f : 0.0f; }
__CPUSIM_DEV float __cpusim_round_f(double d, double residual, int mode) {
    if (__builtin_isinf(d) || d != d) return (float)d;
    float r = (float)d;
    return __cpusim_dir_f(r, __cpusim_sign_f((d - (double)r) + residual), mode);
}
__CPUSIM_DEV double __cpusim_two_sum_err(double a, double b, double s) {
    double bb = s - a;
    return (a - (s - bb)) + (b - bb);
}
__CPUSIM_DEV float __cpusim_fadd_dir(float a, float b, int mode) {
    double s = (double)a + b;
    /* An exact zero is -0 rounding down, unless both operands are +0. */
    if (s == 0 && mode == 2) return (__builtin_signbit(a) || __builtin_signbit(b) || a != 0) ? -0.0f : 0.0f;
    return __cpusim_round_f(s, __cpusim_two_sum_err(a, b, s), mode);
}
__CPUSIM_DEV float __cpusim_fmul_dir(float a, float b, int mode) { return __cpusim_round_f((double)a * b, 0.0, mode); }
__CPUSIM_DEV float __cpusim_fdiv_dir(float a, float b, int mode) {
    double q = (double)a / b;
    if (__builtin_isinf(q) || q != q || q == 0) return (float)q;
    float r = (float)q;
    if (__builtin_isinf(r)) return __cpusim_dir_f(r, 1.0f, mode);
    double rem = (double)a - (double)r * b;  /* exact: r * b fits in a double */
    return __cpusim_dir_f(r, __cpusim_sign_f(b < 0 ? -rem : rem), mode);
}
__CPUSIM_DEV float __cpusim_fsqrt_dir(float a, int mode) {
    if (!(a > 0) || __builtin_isinf(a)) return __builtin_sqrtf(a);
    float r = (float)__builtin_sqrt((double)a);
    return __cpusim_dir_f(r, __cpusim_sign_f((double)a - (double)r * r), mode);
}
__CPUSIM_DEV float __cpusim_fmaf_dir(float a, float b, float c, int mode) {
    double p = (double)a * b, s = p + c;
    return __cpusim_round_f(s, __cpusim_two_sum_err(p, c, s), mode);
}
#define __CPUSIM_ROUNDED(MODE, M)                                                                               \
    __CPUSIM_DEV float __fadd_##MODE(float x, float y) { return __cpusim_fadd_dir(x, y, M); }                   \
    __CPUSIM_DEV float __fsub_##MODE(float x, float y) { return __cpusim_fadd_dir(x, -y, M); }                  \
    __CPUSIM_DEV float __fmul_##MODE(float x, float y) { return __cpusim_fmul_dir(x, y, M); }                   \
    __CPUSIM_DEV float __fdiv_##MODE(float x, float y) { return __cpusim_fdiv_dir(x, y, M); }                   \
    __CPUSIM_DEV float __frcp_##MODE(float x) { return __cpusim_fdiv_dir(1.0f, x, M); }                         \
    __CPUSIM_DEV float __fsqrt_##MODE(float x) { return __cpusim_fsqrt_dir(x, M); }                             \
    __CPUSIM_DEV float __fmaf_##MODE(float x, float y, float z) { return __cpusim_fmaf_dir(x, y, z, M); }       \
    __CPUSIM_DEV double __dadd_##MODE(double x, double y) { return __cpusim_add_(x, y, M); }                    \
    __CPUSIM_DEV double __dsub_##MODE(double x, double y) { return __cpusim_add_(x, -y, M); }                   \
    __CPUSIM_DEV double __dmul_##MODE(double x, double y) { return __cpusim_mul_(x, y, M); }                    \
    __CPUSIM_DEV double __ddiv_##MODE(double x, double y) { return __cpusim_div_(x, y, M); }                    \
    __CPUSIM_DEV double __drcp_##MODE(double x) { return __cpusim_div_(1.0, x, M); }                            \
    __CPUSIM_DEV double __dsqrt_##MODE(double x) { return __cpusim_sqrt_(x, M); }                               \
    __CPUSIM_DEV double __fma_##MODE(double x, double y, double z) { return __cpusim_fma_(x, y, z, M); }        \
    __CPUSIM_DEV float __int2float_##MODE(int x) { return __cpusim_round_f((double)x, 0.0, M); } \
    __CPUSIM_DEV float __uint2float_##MODE(unsigned x) { return __cpusim_round_f((double)x, 0.0, M); }           \
    __CPUSIM_DEV float __ll2float_##MODE(long long x) {                                                          \
        return __cpusim_dir_f((float)x, __cpusim_sign_f((double)((__int128)x - (__int128)(float)x)), M);         \
    }                                                                                                            \
    __CPUSIM_DEV float __ull2float_##MODE(unsigned long long x) {                                                \
        return __cpusim_dir_f((float)x, __cpusim_sign_f((double)((__int128)x - (__int128)(float)x)), M);         \
    }                                                                                                            \
    __CPUSIM_DEV double __ll2double_##MODE(long long x) {                                                        \
        return __cpusim_dir_((double)x, (double)((__int128)x - (__int128)(double)x), M);                         \
    }                                                                                                            \
    __CPUSIM_DEV double __ull2double_##MODE(unsigned long long x) {                                              \
        return __cpusim_dir_((double)x, (double)((__int128)x - (__int128)(double)x), M);                         \
    }                                                                                                            \
    __CPUSIM_DEV float __double2float_##MODE(double x) { return __cpusim_round_f(x, 0.0, M); }
__CPUSIM_ROUNDED(rz, 0)
__CPUSIM_ROUNDED(ru, 1)
__CPUSIM_ROUNDED(rd, 2)
#undef __CPUSIM_ROUNDED
__CPUSIM_DEV double __dsub_rn(double x, double y) { return x - y; }
__CPUSIM_DEV double __ddiv_rn(double x, double y) { return x / y; }
__CPUSIM_DEV double __drcp_rn(double x) { return 1.0 / x; }
__CPUSIM_DEV double __dsqrt_rn(double x) { return __builtin_sqrt(x); }
__CPUSIM_DEV float __ull2float_rn(unsigned long long x) { return (float)x; }
__CPUSIM_DEV double __ll2double_rn(long long x) { return (double)x; }
__CPUSIM_DEV double __ull2double_rn(unsigned long long x) { return (double)x; }
__CPUSIM_DEV double __uint2double_rn(unsigned x) { return (double)x; }
__CPUSIM_DEV long long __float2ll_rz(float x) { return (long long)x; }
__CPUSIM_DEV long long __float2ll_rd(float x) { return (long long)__builtin_floorf(x); }
__CPUSIM_DEV long long __float2ll_ru(float x) { return (long long)__builtin_ceilf(x); }
__CPUSIM_DEV unsigned __float2uint_rd(float x) { return (unsigned)__builtin_floorf(x); }
__CPUSIM_DEV unsigned __float2uint_ru(float x) { return (unsigned)__builtin_ceilf(x); }
__CPUSIM_DEV unsigned long long __float2ull_rn(float x) { return (unsigned long long)__builtin_rintf(x); }
__CPUSIM_DEV unsigned long long __float2ull_rz(float x) { return (unsigned long long)x; }
__CPUSIM_DEV unsigned long long __float2ull_rd(float x) { return (unsigned long long)__builtin_floorf(x); }
__CPUSIM_DEV unsigned long long __float2ull_ru(float x) { return (unsigned long long)__builtin_ceilf(x); }
__CPUSIM_DEV int __double2int_rd(double x) { return (int)__builtin_floor(x); }
__CPUSIM_DEV int __double2int_ru(double x) { return (int)__builtin_ceil(x); }
__CPUSIM_DEV unsigned __double2uint_rn(double x) { return (unsigned)__builtin_rint(x); }
__CPUSIM_DEV unsigned __double2uint_rz(double x) { return (unsigned)x; }
__CPUSIM_DEV unsigned __double2uint_rd(double x) { return (unsigned)__builtin_floor(x); }
__CPUSIM_DEV unsigned __double2uint_ru(double x) { return (unsigned)__builtin_ceil(x); }
__CPUSIM_DEV long long __double2ll_rn(double x) { return (long long)__builtin_rint(x); }
__CPUSIM_DEV long long __double2ll_rz(double x) { return (long long)x; }
__CPUSIM_DEV long long __double2ll_rd(double x) { return (long long)__builtin_floor(x); }
__CPUSIM_DEV long long __double2ll_ru(double x) { return (long long)__builtin_ceil(x); }
__CPUSIM_DEV unsigned long long __double2ull_rn(double x) { return (unsigned long long)__builtin_rint(x); }
__CPUSIM_DEV unsigned long long __double2ull_rz(double x) { return (unsigned long long)x; }

/* Rounded averages, clamped funnel shifts. */
__CPUSIM_DEV int __rhadd(int x, int y) { return (int)(((long long)x + y + 1) >> 1); }
__CPUSIM_DEV unsigned __urhadd(unsigned x, unsigned y) { return (unsigned)(((unsigned long long)x + y + 1) >> 1); }
__CPUSIM_DEV unsigned __funnelshift_lc(unsigned lo, unsigned hi, unsigned s) {
    return (unsigned)((((unsigned long long)hi << 32 | lo) << (s > 32 ? 32 : s)) >> 32);
}
__CPUSIM_DEV unsigned __funnelshift_rc(unsigned lo, unsigned hi, unsigned s) {
    return (unsigned)(((unsigned long long)hi << 32 | lo) >> (s > 32 ? 32 : s));
}
__CPUSIM_DEV long long llabs(long long x) { return x < 0 ? -x : x; }
__CPUSIM_DEV long labs(long x) { return x < 0 ? -x : x; }

/* More of the math library. Functions with no vector version (lgamma,
 * Bessel, nextafter...) become calls of the C library's, once per lane. */
#define __CPUSIM_LIBM1(NAME)                                                                   \
    __CPUSIM_DEV float NAME##f(float x) { return (float)__builtin_##NAME((double)x); }         \
    __CPUSIM_DEV double NAME(double x) { return __builtin_##NAME(x); }                         \
    __CPUSIM_DEV float NAME(float x) { return (float)__builtin_##NAME((double)x); }
__CPUSIM_LIBM1(lgamma) __CPUSIM_LIBM1(tgamma) __CPUSIM_LIBM1(logb)
#undef __CPUSIM_LIBM1
/* The Bessel functions have no builtin: the C library's symbols by name. */
extern "C" {
/* (glibc's: up to ~10 ulp near their zeros, above CUDA's 7 for j0 and y0.) */
__device__ double __cpusim_j0(double) __asm__("j0");
__device__ double __cpusim_j1(double) __asm__("j1");
__device__ double __cpusim_y0(double) __asm__("y0");
__device__ double __cpusim_y1(double) __asm__("y1");
__device__ double __cpusim_jn(int, double) __asm__("jn");
__device__ double __cpusim_yn(int, double) __asm__("yn");
}
#define __CPUSIM_BESSEL(NAME)                                                                   \
    __CPUSIM_DEV double NAME(double x) { return __cpusim_##NAME(x); }                          \
    __CPUSIM_DEV float NAME##f(float x) { return (float)__cpusim_##NAME((double)x); }          \
    __CPUSIM_DEV float NAME(float x) { return (float)__cpusim_##NAME((double)x); }
__CPUSIM_BESSEL(j0) __CPUSIM_BESSEL(j1) __CPUSIM_BESSEL(y0) __CPUSIM_BESSEL(y1)
#undef __CPUSIM_BESSEL
__CPUSIM_DEV double jn(int n, double x) { return __cpusim_jn(n, x); }
__CPUSIM_DEV float jnf(int n, float x) { return (float)__cpusim_jn(n, (double)x); }
__CPUSIM_DEV double yn(int n, double x) { return __cpusim_yn(n, x); }
__CPUSIM_DEV float ynf(int n, float x) { return (float)__cpusim_yn(n, (double)x); }
__CPUSIM_DEV int ilogbf(float x) { return __builtin_ilogbf(x); }
__CPUSIM_DEV int ilogb(double x) { return __builtin_ilogb(x); }
__CPUSIM_DEV int ilogb(float x) { return __builtin_ilogbf(x); }
__CPUSIM_DEV float ldexpf(float x, int e) { return __builtin_ldexpf(x, e); }
__CPUSIM_DEV double ldexp(double x, int e) { return __builtin_ldexp(x, e); }
__CPUSIM_DEV float ldexp(float x, int e) { return __builtin_ldexpf(x, e); }
__CPUSIM_DEV float scalbnf(float x, int e) { return __builtin_ldexpf(x, e); }
__CPUSIM_DEV double scalbn(double x, int e) { return __builtin_ldexp(x, e); }
__CPUSIM_DEV float scalbn(float x, int e) { return __builtin_ldexpf(x, e); }
__CPUSIM_DEV float scalblnf(float x, long e) { return __builtin_ldexpf(x, (int)(e > 1000 ? 1000 : e < -1000 ? -1000 : e)); }
__CPUSIM_DEV float nextafterf(float x, float y) { return __builtin_nextafterf(x, y); }
__CPUSIM_DEV double nextafter(double x, double y) { return __builtin_nextafter(x, y); }
__CPUSIM_DEV float nextafter(float x, float y) { return __builtin_nextafterf(x, y); }
__CPUSIM_DEV float remainderf(float x, float y) { return (float)__builtin_remainder((double)x, (double)y); }
__CPUSIM_DEV double remainder(double x, double y) { return __builtin_remainder(x, y); }
__CPUSIM_DEV float remainder(float x, float y) { return (float)__builtin_remainder((double)x, (double)y); }
extern "C" __device__ double __cpusim_remquo(double, double, int*) __asm__("remquo");
__CPUSIM_DEV double remquo(double x, double y, int* q) { return __cpusim_remquo(x, y, q); }
__CPUSIM_DEV float remquof(float x, float y, int* q) { return (float)remquo((double)x, (double)y, q); }
__CPUSIM_DEV float remquo(float x, float y, int* q) { return remquof(x, y, q); }
__CPUSIM_DEV long lrintf(float x) { return __builtin_lrintf(x); }
__CPUSIM_DEV long lrint(double x) { return __builtin_lrint(x); }
__CPUSIM_DEV long long llrintf(float x) { return __builtin_llrintf(x); }
__CPUSIM_DEV long long llrint(double x) { return __builtin_llrint(x); }
__CPUSIM_DEV long lroundf(float x) { return __builtin_lroundf(x); }
__CPUSIM_DEV long lround(double x) { return __builtin_lround(x); }
__CPUSIM_DEV long long llroundf(float x) { return __builtin_llroundf(x); }
__CPUSIM_DEV long long llround(double x) { return __builtin_llround(x); }
/* frexp and modf without the C library (their pointer would be a lane's). */
__CPUSIM_DEV double frexp(double x, int* e) {
    if (x == 0 || x != x || __builtin_isinf(x)) {
        *e = 0;
        return x;
    }
    int k = __builtin_ilogb(x) + 1;
    *e = k;
    return __builtin_ldexp(x, -k);
}
__CPUSIM_DEV float frexpf(float x, int* e) {
    if (x == 0 || x != x || __builtin_isinf(x)) {
        *e = 0;
        return x;
    }
    int k = __builtin_ilogbf(x) + 1;
    *e = k;
    return __builtin_ldexpf(x, -k);
}
__CPUSIM_DEV float frexp(float x, int* e) { return frexpf(x, e); }
__CPUSIM_DEV double modf(double x, double* i) {
    double t = __builtin_trunc(x);
    *i = t;
    return __builtin_isinf(x) ? __builtin_copysign(0.0, x) : __builtin_copysign(x - t, x);
}
__CPUSIM_DEV float modff(float x, float* i) {
    float t = __builtin_truncf(x);
    *i = t;
    return __builtin_isinf(x) ? __builtin_copysignf(0.0f, x) : __builtin_copysignf(x - t, x);
}
__CPUSIM_DEV float modf(float x, float* i) { return modff(x, i); }
/* Norms: in double, so no intermediate overflows. */
/* Sums of squares as two doubles (each square exact with an fma), scaled
 * by a power of two so nothing overflows, then a square root corrected by
 * one Newton step. */
__CPUSIM_DEV void __cpusim_sum_squares(double a, double b, double c, double* hi, double* lo, int* e) {
    double m = __builtin_fmax(__builtin_fabs(a), __builtin_fmax(__builtin_fabs(b), __builtin_fabs(c)));
    *e = m > 0 ? __builtin_ilogb(m) : 0;
    a = __builtin_ldexp(a, -*e), b = __builtin_ldexp(b, -*e), c = __builtin_ldexp(c, -*e);
    double s = a * a, err = __builtin_fma(a, a, -s);
#define __CPUSIM_ADD_SQUARE(v)                                  \
    {                                                           \
        double p = v * v, pe = __builtin_fma(v, v, -p);         \
        double t = s + p, bb = t - s;                           \
        err += (s - (t - bb)) + (p - bb) + pe;                  \
        s = t;                                                  \
    }
    __CPUSIM_ADD_SQUARE(b) __CPUSIM_ADD_SQUARE(c)
#undef __CPUSIM_ADD_SQUARE
    *hi = s + err;
    *lo = err - (*hi - s);
}
__CPUSIM_DEV double norm3d(double a, double b, double c) {
    if (__builtin_isinf(a) || __builtin_isinf(b) || __builtin_isinf(c)) return __builtin_inf();
    if (a != a || b != b || c != c) return a + b + c;
    double hi, lo;
    int e;
    __cpusim_sum_squares(a, b, c, &hi, &lo, &e);
    if (hi == 0) return 0.0;
    double r = __builtin_sqrt(hi);
    r += (__builtin_fma(-r, r, hi) + lo) / (2.0 * r);
    return __builtin_ldexp(r, e);
}
__CPUSIM_DEV float norm3df(float a, float b, float c) {
    return (float)__builtin_sqrt((double)a * a + (double)b * b + (double)c * c);
}
__CPUSIM_DEV double rnorm3d(double a, double b, double c) {
    if (__builtin_isinf(a) || __builtin_isinf(b) || __builtin_isinf(c)) return 0.0;
    if (a != a || b != b || c != c) return a + b + c;
    double hi, lo;
    int e;
    __cpusim_sum_squares(a, b, c, &hi, &lo, &e);
    if (hi == 0) return __builtin_inf();
    double y = 1.0 / __builtin_sqrt(hi);
    /* y += y (1 - s y^2) / 2, with s y^2 from the two-double s */
    double y2 = y * y, e2 = __builtin_fma(y, y, -y2);
    double sy2 = hi * y2, es = __builtin_fma(hi, y2, -sy2) + hi * e2 + lo * y2;
    y += 0.5 * y * ((1.0 - sy2) - es);
    return __builtin_ldexp(y, -e);
}
__CPUSIM_DEV float rnorm3df(float a, float b, float c) {
    return (float)(1.0 / __builtin_sqrt((double)a * a + (double)b * b + (double)c * c));
}
__CPUSIM_DEV double norm4d(double a, double b, double c, double d) { return __builtin_hypot(__builtin_hypot(a, b), __builtin_hypot(c, d)); }
__CPUSIM_DEV float norm4df(float a, float b, float c, float d) {
    return (float)__builtin_sqrt((double)a * a + (double)b * b + (double)c * c + (double)d * d);
}
__CPUSIM_DEV double rnorm4d(double a, double b, double c, double d) { return 1.0 / norm4d(a, b, c, d); }
__CPUSIM_DEV float rnorm4df(float a, float b, float c, float d) { return (float)(1.0 / (double)norm4df(a, b, c, d)); }
__CPUSIM_DEV double rhypot(double a, double b) { return 1.0 / __builtin_hypot(a, b); }
__CPUSIM_DEV float rhypotf(float a, float b) { return (float)(1.0 / __builtin_hypot((double)a, (double)b)); }
/* The inverse error functions: Giles' approximation, then two Newton steps
 * on erf in double. */
__CPUSIM_DEV double erfinv(double y) {
    if (y <= -1.0) return y == -1.0 ? -__builtin_inf() : __builtin_nan("");
    if (y >= 1.0) return y == 1.0 ? __builtin_inf() : __builtin_nan("");
    double w = -__builtin_log((1.0 - y) * (1.0 + y)), x;
    if (w < 6.25) {
        w -= 3.125;
        x = -3.6444120640178196996e-21;
        x = -1.685059138182016589e-19 + x * w; x = 1.2858480715256400167e-18 + x * w;
        x = 1.115787767802518096e-17 + x * w; x = -1.333171662854620906e-16 + x * w;
        x = 2.0972767875968561637e-17 + x * w; x = 6.6376381343583238325e-15 + x * w;
        x = -4.0545662729752068639e-14 + x * w; x = -8.1519341976054721522e-14 + x * w;
        x = 2.6335093153082322977e-12 + x * w; x = -1.2975133253453532498e-11 + x * w;
        x = -5.4154120542946279317e-11 + x * w; x = 1.051212273321532285e-09 + x * w;
        x = -4.1126339803469836976e-09 + x * w; x = -2.9070369957882005086e-08 + x * w;
        x = 4.2347877827932403518e-07 + x * w; x = -1.3654692000834678645e-06 + x * w;
        x = -1.3882523362786468719e-05 + x * w; x = 0.0001867342080340571352 + x * w;
        x = -0.00074070253416626697512 + x * w; x = -0.0060336708714301490533 + x * w;
        x = 0.24015818242558961693 + x * w; x = 1.6536545626831027356 + x * w;
    } else if (w < 16.0) {
        w = __builtin_sqrt(w) - 3.25;
        x = 2.2137376921775787049e-09;
        x = 9.0756561938885390979e-08 + x * w; x = -2.7517406297064545428e-07 + x * w;
        x = 1.8239629214389227755e-08 + x * w; x = 1.5027403968909827627e-06 + x * w;
        x = -4.013867526981545969e-06 + x * w; x = 2.9234449089955446044e-06 + x * w;
        x = 1.2475304481671778723e-05 + x * w; x = -4.7318229009055733981e-05 + x * w;
        x = 6.8284851459573175448e-05 + x * w; x = 2.4031110387097893999e-05 + x * w;
        x = -0.0003550375203628474796 + x * w; x = 0.00095328937973738049703 + x * w;
        x = -0.0016882755560235047313 + x * w; x = 0.0024914420961078508066 + x * w;
        x = -0.0037512085075692412107 + x * w; x = 0.005370914553590063617 + x * w;
        x = 1.0052589676941592334 + x * w; x = 3.0838856104922207635 + x * w;
    } else {
        w = __builtin_sqrt(w) - 5.0;
        x = -2.7109920616438573243e-11;
        x = -2.5556418169965252055e-10 + x * w; x = 1.5076572693500548083e-09 + x * w;
        x = -3.7894654401267369937e-09 + x * w; x = 7.6157012080783393804e-09 + x * w;
        x = -1.4960026627149240478e-08 + x * w; x = 2.9147953450901080826e-08 + x * w;
        x = -6.7711997758452339498e-08 + x * w; x = 2.2900482228026654717e-07 + x * w;
        x = -9.9298272942317002539e-07 + x * w; x = 4.5260625972231537039e-06 + x * w;
        x = -1.9681778105531670567e-05 + x * w; x = 7.5995277030017761139e-05 + x * w;
        x = -0.00021503011930044477347 + x * w; x = -0.00013871931833623122026 + x * w;
        x = 1.0103004648645343977 + x * w; x = 4.8499064014085844221 + x * w;
    }
    x *= y;
    for (int i = 0; i < 2; ++i)  /* Newton on erf(x) - y */
        x -= (__builtin_erf(x) - y) / (1.1283791670955126 * __builtin_exp(-x * x));
    return x;
}
__CPUSIM_DEV float erfinvf(float y) { return (float)erfinv((double)y); }
__CPUSIM_DEV double erfcx(double x);
__CPUSIM_DEV double erfcinv(double y) {
    if (y <= 0.0) return y == 0.0 ? __builtin_inf() : __builtin_nan("");
    if (y >= 2.0) return y == 2.0 ? -__builtin_inf() : __builtin_nan("");
    if (y > 0.0625 && y < 1.9375) return erfinv(1.0 - y);
    /* Tails: Newton on log erfc(x) = log erfcx(x) - x^2, which stays
     * representable where erfc itself underflows. */
    double t = y < 1.0 ? y : 2.0 - y, lt = __builtin_log(t);
    double x = __builtin_sqrt(-lt);
    for (int i = 0; i < 60; ++i) {
        double cx = erfcx(x);
        double g = __builtin_log(cx) - x * x - lt;
        double dx = g / (-1.1283791670955126 / cx);
        x -= dx;
        if (__builtin_fabs(dx) <= 1e-16 * __builtin_fabs(x)) break;
    }
    return y < 1.0 ? x : -x;
}
__CPUSIM_DEV float erfcinvf(float y) { return (float)erfcinv((double)y); }
/* exp(x^2) erfc(x): directly where that does not overflow, else the
 * asymptotic series. */
__CPUSIM_DEV double erfcx(double x) {
    if (x < 25.0) return __builtin_exp(x * x) * __builtin_erfc(x);
    /* sum (-1)^n (2n-1)!! / (2x^2)^n, while its terms decrease */
    double u = 0.5 / (x * x), term = 1.0, sum = 1.0;
    for (int n = 1; n < 40; ++n) {
        double next = -term * (2 * n - 1) * u;
        if (__builtin_fabs(next) >= __builtin_fabs(term)) break;
        term = next;
        sum += term;
    }
    return sum / (x * 1.772453850905516);
}
__CPUSIM_DEV float erfcxf(float x) { return (float)erfcx((double)x); }
/* erfc(-x / sqrt 2) with the argument as a sum of two doubles: in the tail
 * its rounding error is amplified by about x^2. */
__CPUSIM_DEV double normcdf(double x) {
    const double h = 0.70710678118654757, l = -4.8336466567264567e-17;
    if (__builtin_isinf(x)) return x > 0 ? 1.0 : 0.0;
    double t = -x * h, tl = __builtin_fma(-x, h, -t) - x * l;
    return 0.5 * (__builtin_erfc(t) - tl * 1.1283791670955126 * __builtin_exp(-t * t));
}
__CPUSIM_DEV float normcdff(float x) { return (float)normcdf((double)x); }
__CPUSIM_DEV double normcdfinv(double p) { return 0.0 - 1.41421356237309504880 * erfcinv(2.0 * p); }
__CPUSIM_DEV float normcdfinvf(float p) { return (float)normcdfinv((double)p); }
/* Modified Bessel functions of order 0 and 1: their power series converge
 * for every x; in double, its terms stay finite as far as the result does. */
__CPUSIM_DEV double cyl_bessel_i0(double x) {
    if (__builtin_fabs(x) > 713.0) return __builtin_inf();
    double q = x * x * 0.25, term = 1.0, sum = 1.0;
    for (int k = 1; k < 2000 && term > 1e-17 * sum; ++k) term *= q / ((double)k * k), sum += term;
    return sum;
}
__CPUSIM_DEV float cyl_bessel_i0f(float x) { return (float)cyl_bessel_i0((double)x); }
__CPUSIM_DEV double cyl_bessel_i1(double x) {
    if (__builtin_fabs(x) > 713.0) return __builtin_copysign(__builtin_inf(), x);
    double q = x * x * 0.25, term = x * 0.5, sum = term;
    for (int k = 1; k < 2000 && __builtin_fabs(term) > 1e-17 * __builtin_fabs(sum); ++k) term *= q / ((double)k * (k + 1)), sum += term;
    return sum;
}
__CPUSIM_DEV float cyl_bessel_i1f(float x) { return (float)cyl_bessel_i1((double)x); }

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

/* Atomics. Each becomes one LLVM atomic instruction per thread (relaxed, as
 * CUDA's); the SIMT compiler turns the warp's into one locked instruction
 * per active lane, or, when every lane hits the same address, one reduction
 * across the lanes and a single locked instruction. The _block and _system
 * scopes are the same thing on a CPU. */
#define __CPUSIM_ATOMIC_SCOPES(R, NAME, PARAMS, ARGS)                         \
    __CPUSIM_DEV R NAME##_block PARAMS { return NAME ARGS; }                  \
    __CPUSIM_DEV R NAME##_system PARAMS { return NAME ARGS; }
#define __CPUSIM_ATOMIC_RMW(NAME, BUILTIN, T)                                                     \
    __CPUSIM_DEV T NAME(T* p, T v) { return BUILTIN(p, v, __ATOMIC_RELAXED); }                   \
    __CPUSIM_ATOMIC_SCOPES(T, NAME, (T * p, T v), (p, v))
__CPUSIM_ATOMIC_RMW(atomicAdd, __atomic_fetch_add, int)
__CPUSIM_ATOMIC_RMW(atomicAdd, __atomic_fetch_add, unsigned int)
__CPUSIM_ATOMIC_RMW(atomicAdd, __atomic_fetch_add, unsigned long long)
__CPUSIM_ATOMIC_RMW(atomicAdd, __atomic_fetch_add, float)
__CPUSIM_ATOMIC_RMW(atomicAdd, __atomic_fetch_add, double)
__CPUSIM_ATOMIC_RMW(atomicSub, __atomic_fetch_sub, int)
__CPUSIM_ATOMIC_RMW(atomicSub, __atomic_fetch_sub, unsigned int)
__CPUSIM_ATOMIC_RMW(atomicMin, __atomic_fetch_min, int)
__CPUSIM_ATOMIC_RMW(atomicMin, __atomic_fetch_min, unsigned int)
__CPUSIM_ATOMIC_RMW(atomicMin, __atomic_fetch_min, long long)
__CPUSIM_ATOMIC_RMW(atomicMin, __atomic_fetch_min, unsigned long long)
__CPUSIM_ATOMIC_RMW(atomicMax, __atomic_fetch_max, int)
__CPUSIM_ATOMIC_RMW(atomicMax, __atomic_fetch_max, unsigned int)
__CPUSIM_ATOMIC_RMW(atomicMax, __atomic_fetch_max, long long)
__CPUSIM_ATOMIC_RMW(atomicMax, __atomic_fetch_max, unsigned long long)
__CPUSIM_ATOMIC_RMW(atomicAnd, __atomic_fetch_and, int)
__CPUSIM_ATOMIC_RMW(atomicAnd, __atomic_fetch_and, unsigned int)
__CPUSIM_ATOMIC_RMW(atomicAnd, __atomic_fetch_and, unsigned long long)
__CPUSIM_ATOMIC_RMW(atomicOr, __atomic_fetch_or, int)
__CPUSIM_ATOMIC_RMW(atomicOr, __atomic_fetch_or, unsigned int)
__CPUSIM_ATOMIC_RMW(atomicOr, __atomic_fetch_or, unsigned long long)
__CPUSIM_ATOMIC_RMW(atomicXor, __atomic_fetch_xor, int)
__CPUSIM_ATOMIC_RMW(atomicXor, __atomic_fetch_xor, unsigned int)
__CPUSIM_ATOMIC_RMW(atomicXor, __atomic_fetch_xor, unsigned long long)
#undef __CPUSIM_ATOMIC_RMW
#define __CPUSIM_ATOMIC_EXCH(T)                                                                   \
    __CPUSIM_DEV T atomicExch(T* p, T v) {                                                        \
        T r;                                                                                      \
        __atomic_exchange(p, &v, &r, __ATOMIC_RELAXED);                                           \
        return r;                                                                                 \
    }                                                                                             \
    __CPUSIM_ATOMIC_SCOPES(T, atomicExch, (T * p, T v), (p, v))
__CPUSIM_ATOMIC_EXCH(int)
__CPUSIM_ATOMIC_EXCH(unsigned int)
__CPUSIM_ATOMIC_EXCH(unsigned long long)
__CPUSIM_ATOMIC_EXCH(float)
#undef __CPUSIM_ATOMIC_EXCH
#define __CPUSIM_ATOMIC_CAS(T)                                                                    \
    __CPUSIM_DEV T atomicCAS(T* p, T compare, T v) {                                              \
        __atomic_compare_exchange_n(p, &compare, v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);   \
        return compare;                                                                           \
    }                                                                                             \
    __CPUSIM_ATOMIC_SCOPES(T, atomicCAS, (T * p, T compare, T v), (p, compare, v))
__CPUSIM_ATOMIC_CAS(int)
__CPUSIM_ATOMIC_CAS(unsigned int)
__CPUSIM_ATOMIC_CAS(unsigned long long)
__CPUSIM_ATOMIC_CAS(unsigned short)
#undef __CPUSIM_ATOMIC_CAS
/* ((old >= v) ? 0 : old + 1) and (((old == 0) || (old > v)) ? v : old - 1):
 * LLVM's uinc_wrap and udec_wrap, which the compiler makes of these. */
__CPUSIM_DEV unsigned int atomicInc(unsigned int* p, unsigned int v) { return __nvvm_atom_inc_gen_ui(p, v); }
__CPUSIM_DEV unsigned int atomicDec(unsigned int* p, unsigned int v) { return __nvvm_atom_dec_gen_ui(p, v); }
__CPUSIM_ATOMIC_SCOPES(unsigned int, atomicInc, (unsigned int* p, unsigned int v), (p, v))
__CPUSIM_ATOMIC_SCOPES(unsigned int, atomicDec, (unsigned int* p, unsigned int v), (p, v))
#undef __CPUSIM_ATOMIC_SCOPES

/* Barriers that also vote: the number of threads with a nonzero predicate,
 * and whether all or any have one (cpusim-devc lowers them to a counter in
 * shared memory around an ordinary barrier). */
__CPUSIM_DEV int __syncthreads_count(int pred) { return __nvvm_bar0_popc(pred); }
__CPUSIM_DEV int __syncthreads_and(int pred) { return __nvvm_bar0_and(pred); }
__CPUSIM_DEV int __syncthreads_or(int pred) { return __nvvm_bar0_or(pred); }

/* printf, malloc and free in device code, as declared by Clang's own CUDA
 * headers. Clang turns printf into vprintf(format, packed arguments), which
 * cpusim-devc rewrites into one call per argument; malloc and free are the
 * process's own (the driver's), called once per active lane. */
extern "C" {
__device__ int printf(const char*, ...);
__device__ int vprintf(const char*, const char*);
__device__ void* malloc(size_t) __attribute__((nothrow, malloc));
__device__ void free(void*) __attribute__((nothrow));
__device__ void* memset(void*, int, size_t);
__device__ void* memcpy(void*, const void*, size_t);
/* assert(): the message, then the kernel stops; the launch fails with
 * cudaErrorAssert. */
__device__ static __inline__ void __assert_fail(const char* expr, const char* file, unsigned int line,
                                                const char* function) {
    printf("%s:%u: %s: Assertion `%s' failed.\n", file, line, function, expr);
    __builtin_trap();
}
}

/* Memory fences: a full fence (mfence) for all three scopes. */
__CPUSIM_DEV void __threadfence(void) { __nvvm_membar_gl(); }
__CPUSIM_DEV void __threadfence_block(void) { __nvvm_membar_cta(); }
__CPUSIM_DEV void __threadfence_system(void) { __nvvm_membar_sys(); }
#undef __CPUSIM_DEV

/* new and delete in device code: Clang's own wrapper of <new> defines them
 * on the device malloc and free declared above. */
#include <new>

#endif
