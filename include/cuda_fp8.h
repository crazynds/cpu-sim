/* 8-bit floating point (OCP FP8), CUDA's API: E4M3 (bias 7, no infinity,
 * largest 448) and E5M2 (bias 15, with infinity, largest 57344), the
 * __nv_cvt_* conversions and the __nv_fp8_e4m3 / __nv_fp8_e5m2 classes and
 * their pairs and quadruples. Conversions to fp8 round to nearest even; out
 * of range they give NaN (E4M3) or infinity (E5M2), or with __NV_SATFINITE
 * the largest finite value. Every fp8 value is exact in half, bfloat16 and
 * float. Usable on host and device. */
#ifndef CPUSIM_CUDA_FP8_H
#define CPUSIM_CUDA_FP8_H
#define __CUDA_FP8_TYPES_EXIST__

#include "cuda_bf16.h"
#include "cuda_fp16.h"

typedef unsigned char __nv_fp8_storage_t;
typedef unsigned short __nv_fp8x2_storage_t;
typedef unsigned int __nv_fp8x4_storage_t;

typedef enum __nv_saturation_t { __NV_NOSAT, __NV_SATFINITE } __nv_saturation_t;
typedef enum __nv_fp8_interpretation_t { __NV_E4M3, __NV_E5M2 } __nv_fp8_interpretation_t;

namespace __cpusim_fp8 {

__host__ __device__ inline double pow2(int k) {  // k within double's normal range
    unsigned long long bits = (unsigned long long)(k + 1023) << 52;
    double d;
    __builtin_memcpy(&d, &bits, 8);
    return d;
}

__host__ __device__ inline __nv_fp8_storage_t from_double(double x, __nv_saturation_t sat,
                                                          __nv_fp8_interpretation_t fmt) {
    const bool e4m3 = fmt == __NV_E4M3;
    const int m = e4m3 ? 3 : 2, bias = e4m3 ? 7 : 15;
    const unsigned max_code = e4m3 ? 0x7e : 0x7b;
    const unsigned nan = 0x7f, inf = 0x7c;
    unsigned long long u;
    __builtin_memcpy(&u, &x, 8);
    const unsigned sign = unsigned(u >> 63) << 7;
    const unsigned long long a = u & 0x7fffffffffffffffull;
    if (a > 0x7ff0000000000000ull) return (__nv_fp8_storage_t)(sign | nan);
    if (a == 0x7ff0000000000000ull)
        return (__nv_fp8_storage_t)(sign | (sat == __NV_SATFINITE ? max_code : e4m3 ? nan : inf));
    const int exp = int(a >> 52);
    if (exp == 0) return (__nv_fp8_storage_t)sign;  // a double subnormal: far below fp8's smallest
    // The step between neighbours at this magnitude (subnormal below 2^(1 - bias)).
    int e = exp - 1023;
    if (e < 1 - bias) e = 1 - bias;
    int q = e - m;
    double v;
    __builtin_memcpy(&v, &a, 8);
    double n = __builtin_rint(v * pow2(-q));  // round to nearest even
    if (n == 0.0) return (__nv_fp8_storage_t)sign;
    if (n >= pow2(m + 1)) n *= 0.5, ++q;  // rounded up into the next binade
    unsigned code;
    if (n < pow2(m)) {
        code = unsigned(n);  // subnormal (q is 1 - bias - m)
    } else {
        const int field = q + m + bias;
        code = field > 31 ? 0xffu : unsigned(field << m) | (unsigned(n) - (1u << m));
    }
    if (code > max_code) code = sat == __NV_SATFINITE ? max_code : e4m3 ? nan : inf;
    return (__nv_fp8_storage_t)(sign | code);
}

__host__ __device__ inline float to_float(__nv_fp8_storage_t x, __nv_fp8_interpretation_t fmt) {
    const bool e4m3 = fmt == __NV_E4M3;
    const int m = e4m3 ? 3 : 2, bias = e4m3 ? 7 : 15;
    const float s = (x & 0x80) ? -1.0f : 1.0f;
    const unsigned mag = x & 0x7f;
    const unsigned field = mag >> m, mant = mag & ((1u << m) - 1);
    if (e4m3 && mag == 0x7f) return __builtin_nanf("");
    if (!e4m3 && field == 31) return mant ? __builtin_nanf("") : s * __builtin_inff();
    if (field == 0) return s * float(double(mant) * pow2(1 - bias - m));
    return s * float(double((1u << m) | mant) * pow2(int(field) - bias - m));
}

}  // namespace __cpusim_fp8

/* ---------------------------------------------------------- conversions */
static inline __host__ __device__ __nv_fp8_storage_t __nv_cvt_double_to_fp8(double x, __nv_saturation_t sat,
                                                                            __nv_fp8_interpretation_t fmt) {
    return __cpusim_fp8::from_double(x, sat, fmt);
}
static inline __host__ __device__ __nv_fp8_storage_t __nv_cvt_float_to_fp8(float x, __nv_saturation_t sat,
                                                                           __nv_fp8_interpretation_t fmt) {
    return __cpusim_fp8::from_double((double)x, sat, fmt);
}
static inline __host__ __device__ __nv_fp8x2_storage_t __nv_cvt_double2_to_fp8x2(double2 x, __nv_saturation_t sat,
                                                                                 __nv_fp8_interpretation_t fmt) {
    return (__nv_fp8x2_storage_t)(__nv_cvt_double_to_fp8(x.x, sat, fmt) | (__nv_cvt_double_to_fp8(x.y, sat, fmt) << 8));
}
static inline __host__ __device__ __nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(float2 x, __nv_saturation_t sat,
                                                                                __nv_fp8_interpretation_t fmt) {
    return (__nv_fp8x2_storage_t)(__nv_cvt_float_to_fp8(x.x, sat, fmt) | (__nv_cvt_float_to_fp8(x.y, sat, fmt) << 8));
}
static inline __host__ __device__ __nv_fp8_storage_t __nv_cvt_halfraw_to_fp8(__half_raw x, __nv_saturation_t sat,
                                                                             __nv_fp8_interpretation_t fmt) {
    return __nv_cvt_float_to_fp8(__half2float(__half(x)), sat, fmt);
}
static inline __host__ __device__ __nv_fp8x2_storage_t __nv_cvt_halfraw2_to_fp8x2(__half2_raw x, __nv_saturation_t sat,
                                                                                  __nv_fp8_interpretation_t fmt) {
    return (__nv_fp8x2_storage_t)(__nv_cvt_halfraw_to_fp8(__half_raw{x.x}, sat, fmt) |
                                  (__nv_cvt_halfraw_to_fp8(__half_raw{x.y}, sat, fmt) << 8));
}
static inline __host__ __device__ __nv_fp8_storage_t __nv_cvt_bfloat16raw_to_fp8(__nv_bfloat16_raw x,
                                                                                 __nv_saturation_t sat,
                                                                                 __nv_fp8_interpretation_t fmt) {
    return __nv_cvt_float_to_fp8(__bfloat162float(__nv_bfloat16(x)), sat, fmt);
}
static inline __host__ __device__ __nv_fp8x2_storage_t __nv_cvt_bfloat16raw2_to_fp8x2(__nv_bfloat162_raw x,
                                                                                      __nv_saturation_t sat,
                                                                                      __nv_fp8_interpretation_t fmt) {
    return (__nv_fp8x2_storage_t)(__nv_cvt_bfloat16raw_to_fp8(__nv_bfloat16_raw{x.x}, sat, fmt) |
                                  (__nv_cvt_bfloat16raw_to_fp8(__nv_bfloat16_raw{x.y}, sat, fmt) << 8));
}
static inline __host__ __device__ __half_raw __nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t x, __nv_fp8_interpretation_t fmt) {
    return __half_raw(__half(__cpusim_fp8::to_float(x, fmt)));
}
static inline __host__ __device__ __half2_raw __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t x,
                                                                       __nv_fp8_interpretation_t fmt) {
    __half_raw lo = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)(x & 0xff), fmt);
    __half_raw hi = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)(x >> 8), fmt);
    return __half2_raw{lo.x, hi.x};
}

/* --------------------------------------------------------------- classes */
#define __CPUSIM_FP8_CLASS(NAME, FMT)                                                                    \
    struct __attribute__((aligned(1))) NAME {                                                           \
        __nv_fp8_storage_t __x;                                                                         \
        NAME() = default;                                                                               \
        __host__ __device__ explicit NAME(float f) : __x(__nv_cvt_float_to_fp8(f, __NV_SATFINITE, FMT)) {} \
        __host__ __device__ explicit NAME(double d) : __x(__nv_cvt_double_to_fp8(d, __NV_SATFINITE, FMT)) {} \
        __host__ __device__ explicit NAME(__half h) : NAME(__half2float(h)) {}                           \
        __host__ __device__ explicit NAME(__nv_bfloat16 b) : NAME(__bfloat162float(b)) {}                \
        __host__ __device__ explicit NAME(int v) : NAME((double)v) {}                                     \
        __host__ __device__ explicit NAME(unsigned int v) : NAME((double)v) {}                            \
        __host__ __device__ explicit NAME(long long v) : NAME((double)v) {}                               \
        __host__ __device__ explicit NAME(unsigned long long v) : NAME((double)v) {}                      \
        __host__ __device__ explicit operator float() const { return __cpusim_fp8::to_float(__x, FMT); }   \
        __host__ __device__ explicit operator double() const { return (double)__cpusim_fp8::to_float(__x, FMT); } \
        __host__ __device__ explicit operator __half() const { return __half(__cpusim_fp8::to_float(__x, FMT)); } \
        __host__ __device__ explicit operator __nv_bfloat16() const {                                    \
            return __nv_bfloat16(__cpusim_fp8::to_float(__x, FMT));                                      \
        }                                                                                               \
        __host__ __device__ explicit operator int() const { return (int)__cpusim_fp8::to_float(__x, FMT); } \
        __host__ __device__ explicit operator bool() const { return (__x & 0x7f) != 0; }                 \
    };
__CPUSIM_FP8_CLASS(__nv_fp8_e4m3, __NV_E4M3)
__CPUSIM_FP8_CLASS(__nv_fp8_e5m2, __NV_E5M2)
#undef __CPUSIM_FP8_CLASS

/* Pairs and quadruples: float2 / float4 in, float2 / float4 out. */
#define __CPUSIM_FP8_VEC(NAME, FMT)                                                                        \
    struct __attribute__((aligned(2))) NAME##x2 {                                                          \
        __nv_fp8x2_storage_t __x;                                                                          \
        NAME##x2() = default;                                                                              \
        __host__ __device__ explicit NAME##x2(float2 f) : __x(__nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, FMT)) {} \
        __host__ __device__ explicit operator float2() const {                                            \
            return make_float2(__cpusim_fp8::to_float((__nv_fp8_storage_t)(__x & 0xff), FMT),              \
                               __cpusim_fp8::to_float((__nv_fp8_storage_t)(__x >> 8), FMT));               \
        }                                                                                                  \
    };                                                                                                     \
    struct __attribute__((aligned(4))) NAME##x4 {                                                          \
        __nv_fp8x4_storage_t __x;                                                                          \
        NAME##x4() = default;                                                                              \
        __host__ __device__ explicit NAME##x4(float4 f)                                                    \
            : __x(__nv_cvt_float2_to_fp8x2(make_float2(f.x, f.y), __NV_SATFINITE, FMT) |                   \
                  ((__nv_fp8x4_storage_t)__nv_cvt_float2_to_fp8x2(make_float2(f.z, f.w), __NV_SATFINITE, FMT) << 16)) {} \
        __host__ __device__ explicit operator float4() const {                                            \
            return make_float4(__cpusim_fp8::to_float((__nv_fp8_storage_t)(__x & 0xff), FMT),              \
                               __cpusim_fp8::to_float((__nv_fp8_storage_t)((__x >> 8) & 0xff), FMT),       \
                               __cpusim_fp8::to_float((__nv_fp8_storage_t)((__x >> 16) & 0xff), FMT),      \
                               __cpusim_fp8::to_float((__nv_fp8_storage_t)(__x >> 24), FMT));              \
        }                                                                                                  \
    };
__CPUSIM_FP8_VEC(__nv_fp8_e4m3, __NV_E4M3)
__CPUSIM_FP8_VEC(__nv_fp8_e5m2, __NV_E5M2)
#undef __CPUSIM_FP8_VEC

#endif
