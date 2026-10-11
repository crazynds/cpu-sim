/* Shared by cuda_fp16.h and cuda_bf16.h: bit-level conversions between
 * float/double and the two 16-bit formats, with the rounding modes CUDA names
 * (_rn nearest even, _rz toward zero, _rd down, _ru up).
 *
 * Arithmetic on __half and __nv_bfloat16 is done in float and rounded once to
 * the 16-bit format. For +, -, *, / and sqrt that is the correctly rounded
 * result, as on the GPU: rounding twice (to float, then to p bits) equals
 * rounding once when float's 24 bits are at least 2p + 2 (p = 11 for half,
 * 8 for bfloat16). Fused multiply-add and conversions from double go through
 * float with round-to-odd, which preserves the information the final
 * rounding needs. */
#ifndef CPUSIM_FP16_COMMON_H
#define CPUSIM_FP16_COMMON_H

#if !defined(__CUDA__)  // included from host-only C++
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

#include "vector_types.h"

#define __CPUSIM_HD static __host__ __device__ inline __attribute__((always_inline))

namespace __cpusim_fp {

__CPUSIM_HD unsigned f2u(float f) { return __builtin_bit_cast(unsigned, f); }
__CPUSIM_HD float u2f(unsigned u) { return __builtin_bit_cast(float, u); }

// half <-> float: F16C (vcvtps2ph/vcvtph2ps) on x86.
__CPUSIM_HD unsigned short half_rn(float f) { return __builtin_bit_cast(unsigned short, (_Float16)f); }
__CPUSIM_HD float half_tof(unsigned short h) { return (float)__builtin_bit_cast(_Float16, h); }

// bfloat16 <-> float: the high half of the float's bits, nearest even.
__CPUSIM_HD unsigned short bf16_rn(float f) {
    unsigned u = f2u(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x40u);  // quiet NaN
    return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__CPUSIM_HD float bf16_tof(unsigned short b) { return u2f((unsigned)b << 16); }

enum { RN = 0, RZ = 1, RD = 2, RU = 3 };

// Directed rounding from the nearest-even result r of f: r is within one unit
// of the last place, and both formats are sign-magnitude, so one step on the
// magnitude bits fixes it.
__CPUSIM_HD unsigned short directed(unsigned short r, float f, float back, int mode) {
    if (f != f || mode == RN) return r;
    if (mode == RZ) return __builtin_fabsf(back) > __builtin_fabsf(f) ? (unsigned short)(r - 1) : r;
    if (mode == RD) return back > f ? (unsigned short)((r & 0x8000) ? r + 1 : r - 1) : r;
    return back < f ? (unsigned short)((r & 0x8000) ? r - 1 : r + 1) : r;
}
__CPUSIM_HD unsigned short half_from(float f, int mode) {
    unsigned short r = half_rn(f);
    return directed(r, f, half_tof(r), mode);
}
__CPUSIM_HD unsigned short bf16_from(float f, int mode) {
    unsigned short r = bf16_rn(f);
    return directed(r, f, bf16_tof(r), mode);
}

// double -> float rounded to odd: truncate, then set the last bit if inexact.
__CPUSIM_HD float d2f_odd(double d) {
    float f = (float)d;
    if (d != d) return f;
    if (__builtin_fabs((double)f) > __builtin_fabs(d)) f = u2f(f2u(f) - 1);
    if ((double)f != d) f = u2f(f2u(f) | 1u);
    return f;
}

// x + y rounded to odd in double: the rounded sum and its exact error
// (TwoSum), then, if inexact, the neighbour with an odd last bit. Rounding
// that to fewer bits later gives the correctly rounded exact sum, even when y
// is far below x (a plain double sum would drop it and lose the tie-break).
__CPUSIM_HD double add_odd(double x, double y) {
    double s = x + y;
    if (s - s != 0) return s;  // inf or NaN
    double bv = s - x;
    double e = (x - (s - bv)) + (y - bv);
    unsigned long long u = __builtin_bit_cast(unsigned long long, s);
    if (e != 0 && !(u & 1)) u += (e > 0) == (s > 0) ? 1 : -1;  // one step toward the exact sum
    return __builtin_bit_cast(double, u);
}

// Saturating float -> integer after rounding (NaN gives 0, as in CUDA).
__CPUSIM_HD float round_mode(float v, int mode) {
    return mode == RN ? __builtin_rintf(v) : mode == RZ ? __builtin_truncf(v) : mode == RD ? __builtin_floorf(v) : __builtin_ceilf(v);
}
#define __CPUSIM_SAT(NAME, I, LO, HI, LOF, HIF)                   \
    __CPUSIM_HD I NAME(float v, int mode) {                       \
        v = round_mode(v, mode);                                  \
        if (v != v) return 0;                                     \
        if (v >= (HIF)) return (HI);                              \
        if (v <= (LOF)) return (LO);                              \
        return (I)v;                                              \
    }
__CPUSIM_SAT(to_i8, signed char, -128, 127, -128.0f, 127.0f)
__CPUSIM_SAT(to_u8, unsigned char, 0, 255, 0.0f, 255.0f)
__CPUSIM_SAT(to_i16, short, -32768, 32767, -32768.0f, 32767.0f)
__CPUSIM_SAT(to_u16, unsigned short, 0, 65535, 0.0f, 65535.0f)
__CPUSIM_SAT(to_i32, int, (-2147483647 - 1), 2147483647, -2147483648.0f, 2147483648.0f)
__CPUSIM_SAT(to_u32, unsigned, 0u, 4294967295u, 0.0f, 4294967296.0f)
__CPUSIM_SAT(to_i64, long long, (-9223372036854775807ll - 1), 9223372036854775807ll, -9223372036854775808.0f,
             9223372036854775808.0f)
__CPUSIM_SAT(to_u64, unsigned long long, 0ull, 18446744073709551615ull, 0.0f, 18446744073709551616.0f)
#undef __CPUSIM_SAT

}  // namespace __cpusim_fp

#endif
