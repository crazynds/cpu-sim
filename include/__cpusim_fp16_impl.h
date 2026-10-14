/* The API of one 16-bit float type, included twice: by cuda_fp16.h for
 * __half and by cuda_bf16.h for __nv_bfloat16. The includer defines
 *   __CPUSIM_T, __CPUSIM_T2, __CPUSIM_RAW, __CPUSIM_RAW2   the type names
 *   __CPUSIM_N, __CPUSIM_N2      the name stem in CUDA's function names
 *                                (half/half2, bfloat16/bfloat162)
 *   __CPUSIM_TOF(bits)           exact conversion to float
 *   __CPUSIM_FROM(f, mode)       float -> bits with a rounding mode
 *   __CPUSIM_INF_BITS            the bits of +infinity (larger magnitudes are NaN)
 * No include guard on purpose. */

#define __CPUSIM_CAT_(a, b) a##b
#define __CPUSIM_CAT(a, b) __CPUSIM_CAT_(a, b)
#define __CPUSIM_NAME(a, b) __CPUSIM_CAT(a, b)
#define __CPUSIM_NAME3(a, b, c) __CPUSIM_CAT(__CPUSIM_CAT(a, b), c)
#define __CPUSIM_NAME4(a, b, c, d) __CPUSIM_CAT(__CPUSIM_NAME3(a, b, c), d)

struct __CPUSIM_RAW {
    unsigned short x;
};
struct __CPUSIM_RAW2 {
    unsigned short x, y;
};

struct __attribute__((aligned(2))) __CPUSIM_T {
    unsigned short __x;

    __CPUSIM_T() = default;
    __host__ __device__ constexpr __CPUSIM_T(const __CPUSIM_RAW& r) : __x(r.x) {}
    __host__ __device__ __CPUSIM_T& operator=(const __CPUSIM_RAW& r) { __x = r.x; return *this; }
    __host__ __device__ constexpr operator __CPUSIM_RAW() const { return __CPUSIM_RAW{__x}; }

    __host__ __device__ __CPUSIM_T(float f) : __x(__CPUSIM_FROM(f, __cpusim_fp::RN)) {}
    __host__ __device__ __CPUSIM_T(double d) : __x(__CPUSIM_FROM(__cpusim_fp::d2f_odd(d), __cpusim_fp::RN)) {}
    __host__ __device__ __CPUSIM_T(short v) : __CPUSIM_T((double)v) {}
    __host__ __device__ __CPUSIM_T(unsigned short v) : __CPUSIM_T((double)v) {}
    __host__ __device__ __CPUSIM_T(int v) : __CPUSIM_T((double)v) {}
    __host__ __device__ __CPUSIM_T(unsigned int v) : __CPUSIM_T((double)v) {}
    __host__ __device__ __CPUSIM_T(long v) : __CPUSIM_T((double)v) {}
    __host__ __device__ __CPUSIM_T(unsigned long v) : __CPUSIM_T((double)v) {}
    __host__ __device__ __CPUSIM_T(long long v) : __CPUSIM_T((double)v) {}
    __host__ __device__ __CPUSIM_T(unsigned long long v) : __CPUSIM_T((double)v) {}

    template <typename V>
    __host__ __device__ __CPUSIM_T& operator=(V v) { return *this = __CPUSIM_T(v); }

    __host__ __device__ operator float() const { return __CPUSIM_TOF(__x); }
    // To integers: toward zero, saturating, NaN -> 0.
    __host__ __device__ operator signed char() const { return __cpusim_fp::to_i8(__CPUSIM_TOF(__x), __cpusim_fp::RZ); }
    __host__ __device__ operator unsigned char() const { return __cpusim_fp::to_u8(__CPUSIM_TOF(__x), __cpusim_fp::RZ); }
    __host__ __device__ operator char() const { return (char)__cpusim_fp::to_i8(__CPUSIM_TOF(__x), __cpusim_fp::RZ); }
    __host__ __device__ operator short() const { return __cpusim_fp::to_i16(__CPUSIM_TOF(__x), __cpusim_fp::RZ); }
    __host__ __device__ operator unsigned short() const { return __cpusim_fp::to_u16(__CPUSIM_TOF(__x), __cpusim_fp::RZ); }
    __host__ __device__ operator int() const { return __cpusim_fp::to_i32(__CPUSIM_TOF(__x), __cpusim_fp::RZ); }
    __host__ __device__ operator unsigned int() const { return __cpusim_fp::to_u32(__CPUSIM_TOF(__x), __cpusim_fp::RZ); }
    __host__ __device__ operator long() const { return __cpusim_fp::to_i64(__CPUSIM_TOF(__x), __cpusim_fp::RZ); }
    __host__ __device__ operator unsigned long() const { return __cpusim_fp::to_u64(__CPUSIM_TOF(__x), __cpusim_fp::RZ); }
    __host__ __device__ operator long long() const { return __cpusim_fp::to_i64(__CPUSIM_TOF(__x), __cpusim_fp::RZ); }
    __host__ __device__ operator unsigned long long() const {
        return __cpusim_fp::to_u64(__CPUSIM_TOF(__x), __cpusim_fp::RZ);
    }
    __host__ __device__ constexpr operator bool() const { return (__x & 0x7fffu) != 0; }
};

struct __attribute__((aligned(4))) __CPUSIM_T2 {
    __CPUSIM_T x, y;

    __CPUSIM_T2() = default;
    __host__ __device__ constexpr __CPUSIM_T2(const __CPUSIM_T& a, const __CPUSIM_T& b) : x(a), y(b) {}
    __host__ __device__ constexpr __CPUSIM_T2(const __CPUSIM_RAW2& r) : x(__CPUSIM_RAW{r.x}), y(__CPUSIM_RAW{r.y}) {}
    // A template (exactly the raw type), so that "= {}" is the plain copy
    // assignment instead of a tie between the two.
    template <typename R, int = sizeof(char[__is_same(R, __CPUSIM_RAW2) ? 1 : -1])>
    __host__ __device__ __CPUSIM_T2& operator=(const R& r) { return *this = __CPUSIM_T2(r); }
    __host__ __device__ constexpr operator __CPUSIM_RAW2() const { return __CPUSIM_RAW2{x.__x, y.__x}; }
};

/* ------------------------------------------------------------------ helpers */
namespace __cpusim_fp {
__CPUSIM_HD __CPUSIM_T __CPUSIM_NAME(make_, __CPUSIM_N)(unsigned short bits) { return __CPUSIM_T(__CPUSIM_RAW{bits}); }
__CPUSIM_HD float tof(__CPUSIM_T a) { return __CPUSIM_TOF(a.__x); }
__CPUSIM_HD bool is_nan(__CPUSIM_T a) { return (a.__x & 0x7fffu) > __CPUSIM_INF_BITS; }
}  // namespace __cpusim_fp

#define __CPUSIM_MK(bits) __cpusim_fp::__CPUSIM_NAME(make_, __CPUSIM_N)(bits)
#define __CPUSIM_RN(f) __CPUSIM_MK(__CPUSIM_FROM((f), __cpusim_fp::RN))
#define __CPUSIM_F(a) __cpusim_fp::tof(a)
#define __CPUSIM_CANON_NAN __CPUSIM_MK(0x7fffu)

/* -------------------------------------------------------------- conversions */
__CPUSIM_HD float __CPUSIM_NAME3(__, __CPUSIM_N, 2float)(__CPUSIM_T a) { return __CPUSIM_F(a); }
__CPUSIM_HD __CPUSIM_T __CPUSIM_NAME(__float2, __CPUSIM_N)(float f) { return __CPUSIM_RN(f); }
#define __CPUSIM_FROM_FLOAT(SUFFIX, MODE)                                                  \
    __CPUSIM_HD __CPUSIM_T __CPUSIM_NAME3(__float2, __CPUSIM_N, SUFFIX)(float f) {         \
        return __CPUSIM_MK(__CPUSIM_FROM(f, __cpusim_fp::MODE));                           \
    }
__CPUSIM_FROM_FLOAT(_rn, RN)
__CPUSIM_FROM_FLOAT(_rz, RZ)
__CPUSIM_FROM_FLOAT(_rd, RD)
__CPUSIM_FROM_FLOAT(_ru, RU)
#undef __CPUSIM_FROM_FLOAT
__CPUSIM_HD __CPUSIM_T __CPUSIM_NAME(__double2, __CPUSIM_N)(double d) { return __CPUSIM_T(d); }
__CPUSIM_HD double __CPUSIM_NAME3(__, __CPUSIM_N, 2double)(__CPUSIM_T a) { return (double)__CPUSIM_F(a); }

// Integers: to the type with each rounding mode (saturating), and from it
// (through double, exact for 32-bit values).
#define __CPUSIM_INT_CONV(STEM, I, SAT)                                                                          \
    __CPUSIM_HD I __CPUSIM_NAME4(__, __CPUSIM_N, 2, STEM##_rn)(__CPUSIM_T a) { return __cpusim_fp::SAT(__CPUSIM_F(a), __cpusim_fp::RN); } \
    __CPUSIM_HD I __CPUSIM_NAME4(__, __CPUSIM_N, 2, STEM##_rz)(__CPUSIM_T a) { return __cpusim_fp::SAT(__CPUSIM_F(a), __cpusim_fp::RZ); } \
    __CPUSIM_HD I __CPUSIM_NAME4(__, __CPUSIM_N, 2, STEM##_rd)(__CPUSIM_T a) { return __cpusim_fp::SAT(__CPUSIM_F(a), __cpusim_fp::RD); } \
    __CPUSIM_HD I __CPUSIM_NAME4(__, __CPUSIM_N, 2, STEM##_ru)(__CPUSIM_T a) { return __cpusim_fp::SAT(__CPUSIM_F(a), __cpusim_fp::RU); } \
    __CPUSIM_HD __CPUSIM_T __CPUSIM_NAME4(__##STEM, 2, __CPUSIM_N, _rn)(I v) {                                   \
        return __CPUSIM_MK(__CPUSIM_FROM(__cpusim_fp::d2f_odd((double)v), __cpusim_fp::RN));                    \
    }                                                                                                            \
    __CPUSIM_HD __CPUSIM_T __CPUSIM_NAME4(__##STEM, 2, __CPUSIM_N, _rz)(I v) {                                   \
        return __CPUSIM_MK(__CPUSIM_FROM(__cpusim_fp::d2f_odd((double)v), __cpusim_fp::RZ));                    \
    }                                                                                                            \
    __CPUSIM_HD __CPUSIM_T __CPUSIM_NAME4(__##STEM, 2, __CPUSIM_N, _rd)(I v) {                                   \
        return __CPUSIM_MK(__CPUSIM_FROM(__cpusim_fp::d2f_odd((double)v), __cpusim_fp::RD));                    \
    }                                                                                                            \
    __CPUSIM_HD __CPUSIM_T __CPUSIM_NAME4(__##STEM, 2, __CPUSIM_N, _ru)(I v) {                                   \
        return __CPUSIM_MK(__CPUSIM_FROM(__cpusim_fp::d2f_odd((double)v), __cpusim_fp::RU));                    \
    }
__CPUSIM_INT_CONV(short, short, to_i16)
__CPUSIM_INT_CONV(ushort, unsigned short, to_u16)
__CPUSIM_INT_CONV(int, int, to_i32)
__CPUSIM_INT_CONV(uint, unsigned int, to_u32)
__CPUSIM_INT_CONV(ll, long long, to_i64)
__CPUSIM_INT_CONV(ull, unsigned long long, to_u64)
#undef __CPUSIM_INT_CONV
__CPUSIM_HD signed char __CPUSIM_NAME3(__, __CPUSIM_N, 2char_rz)(__CPUSIM_T a) { return __cpusim_fp::to_i8(__CPUSIM_F(a), __cpusim_fp::RZ); }
__CPUSIM_HD unsigned char __CPUSIM_NAME3(__, __CPUSIM_N, 2uchar_rz)(__CPUSIM_T a) { return __cpusim_fp::to_u8(__CPUSIM_F(a), __cpusim_fp::RZ); }

// Bit reinterpretation.
__CPUSIM_HD short __CPUSIM_NAME3(__, __CPUSIM_N, _as_short)(__CPUSIM_T a) { return (short)a.__x; }
__CPUSIM_HD unsigned short __CPUSIM_NAME3(__, __CPUSIM_N, _as_ushort)(__CPUSIM_T a) { return a.__x; }
__CPUSIM_HD __CPUSIM_T __CPUSIM_NAME(__short_as_, __CPUSIM_N)(short v) { return __CPUSIM_MK((unsigned short)v); }
__CPUSIM_HD __CPUSIM_T __CPUSIM_NAME(__ushort_as_, __CPUSIM_N)(unsigned short v) { return __CPUSIM_MK(v); }

// Pairs.
__CPUSIM_HD __CPUSIM_T2 __CPUSIM_NAME(make_, __CPUSIM_N2)(__CPUSIM_T a, __CPUSIM_T b) { return __CPUSIM_T2(a, b); }
__CPUSIM_HD __CPUSIM_T2 __CPUSIM_NAME(__halves2, __CPUSIM_N2)(__CPUSIM_T a, __CPUSIM_T b) { return __CPUSIM_T2(a, b); }
__CPUSIM_HD __CPUSIM_T2 __CPUSIM_NAME4(__, __CPUSIM_N, 2, __CPUSIM_N2)(__CPUSIM_T a) { return __CPUSIM_T2(a, a); }
__CPUSIM_HD __CPUSIM_T __CPUSIM_NAME(__low2, __CPUSIM_N)(__CPUSIM_T2 a) { return a.x; }
__CPUSIM_HD __CPUSIM_T __CPUSIM_NAME(__high2, __CPUSIM_N)(__CPUSIM_T2 a) { return a.y; }
__CPUSIM_HD __CPUSIM_T2 __CPUSIM_NAME(__low2, __CPUSIM_N2)(__CPUSIM_T2 a) { return __CPUSIM_T2(a.x, a.x); }
__CPUSIM_HD __CPUSIM_T2 __CPUSIM_NAME(__high2, __CPUSIM_N2)(__CPUSIM_T2 a) { return __CPUSIM_T2(a.y, a.y); }
__CPUSIM_HD __CPUSIM_T2 __CPUSIM_NAME(__lows2, __CPUSIM_N2)(__CPUSIM_T2 a, __CPUSIM_T2 b) { return __CPUSIM_T2(a.x, b.x); }
__CPUSIM_HD __CPUSIM_T2 __CPUSIM_NAME(__highs2, __CPUSIM_N2)(__CPUSIM_T2 a, __CPUSIM_T2 b) { return __CPUSIM_T2(a.y, b.y); }
__CPUSIM_HD __CPUSIM_T2 __lowhigh2highlow(__CPUSIM_T2 a) { return __CPUSIM_T2(a.y, a.x); }
__CPUSIM_HD float __low2float(__CPUSIM_T2 a) { return __CPUSIM_F(a.x); }
__CPUSIM_HD float __high2float(__CPUSIM_T2 a) { return __CPUSIM_F(a.y); }
__CPUSIM_HD __CPUSIM_T2 __CPUSIM_NAME3(__float2, __CPUSIM_N2, _rn)(float f) { return __CPUSIM_T2(__CPUSIM_RN(f), __CPUSIM_RN(f)); }
__CPUSIM_HD __CPUSIM_T2 __CPUSIM_NAME3(__floats2, __CPUSIM_N2, _rn)(float a, float b) { return __CPUSIM_T2(__CPUSIM_RN(a), __CPUSIM_RN(b)); }
__CPUSIM_HD __CPUSIM_T2 __CPUSIM_NAME3(__float22, __CPUSIM_N2, _rn)(float2 a) { return __CPUSIM_T2(__CPUSIM_RN(a.x), __CPUSIM_RN(a.y)); }
__CPUSIM_HD float2 __CPUSIM_NAME3(__, __CPUSIM_N2, 2float2)(__CPUSIM_T2 a) { return float2{__CPUSIM_F(a.x), __CPUSIM_F(a.y)}; }

/* --------------------------------------------------------------- arithmetic */
__CPUSIM_HD __CPUSIM_T __hadd(__CPUSIM_T a, __CPUSIM_T b) { return __CPUSIM_RN(__CPUSIM_F(a) + __CPUSIM_F(b)); }
__CPUSIM_HD __CPUSIM_T __hsub(__CPUSIM_T a, __CPUSIM_T b) { return __CPUSIM_RN(__CPUSIM_F(a) - __CPUSIM_F(b)); }
__CPUSIM_HD __CPUSIM_T __hmul(__CPUSIM_T a, __CPUSIM_T b) { return __CPUSIM_RN(__CPUSIM_F(a) * __CPUSIM_F(b)); }
__CPUSIM_HD __CPUSIM_T __hdiv(__CPUSIM_T a, __CPUSIM_T b) { return __CPUSIM_RN(__CPUSIM_F(a) / __CPUSIM_F(b)); }
// The _rn forms only forbid contraction into an fma, which never happens here.
__CPUSIM_HD __CPUSIM_T __hadd_rn(__CPUSIM_T a, __CPUSIM_T b) { return __hadd(a, b); }
__CPUSIM_HD __CPUSIM_T __hsub_rn(__CPUSIM_T a, __CPUSIM_T b) { return __hsub(a, b); }
__CPUSIM_HD __CPUSIM_T __hmul_rn(__CPUSIM_T a, __CPUSIM_T b) { return __hmul(a, b); }
// One rounding of a*b + c: the product is exact in double; the sum is
// rounded to odd in double, then in float, then to nearest.
__CPUSIM_HD __CPUSIM_T __hfma(__CPUSIM_T a, __CPUSIM_T b, __CPUSIM_T c) {
#ifdef __CPUSIM_FMA_HOOK
    return __CPUSIM_MK(__CPUSIM_FMA_HOOK(a.__x, b.__x, c.__x));
#else
    double r = __cpusim_fp::add_odd((double)__CPUSIM_F(a) * (double)__CPUSIM_F(b), (double)__CPUSIM_F(c));
    return __CPUSIM_RN(__cpusim_fp::d2f_odd(r));
#endif
}
__CPUSIM_HD __CPUSIM_T __hneg(__CPUSIM_T a) { return __CPUSIM_MK((unsigned short)(a.__x ^ 0x8000u)); }
__CPUSIM_HD __CPUSIM_T __habs(__CPUSIM_T a) { return __CPUSIM_MK((unsigned short)(a.__x & 0x7fffu)); }
// _sat: clamped to [0, 1], NaN -> +0.
__CPUSIM_HD __CPUSIM_T __cpusim_sat(__CPUSIM_T a) {
    float f = __CPUSIM_F(a);
    return f != f || f <= 0.0f ? __CPUSIM_MK(0) : f >= 1.0f ? __CPUSIM_RN(1.0f) : a;
}
__CPUSIM_HD __CPUSIM_T __hadd_sat(__CPUSIM_T a, __CPUSIM_T b) { return __cpusim_sat(__hadd(a, b)); }
__CPUSIM_HD __CPUSIM_T __hsub_sat(__CPUSIM_T a, __CPUSIM_T b) { return __cpusim_sat(__hsub(a, b)); }
__CPUSIM_HD __CPUSIM_T __hmul_sat(__CPUSIM_T a, __CPUSIM_T b) { return __cpusim_sat(__hmul(a, b)); }
__CPUSIM_HD __CPUSIM_T __hfma_sat(__CPUSIM_T a, __CPUSIM_T b, __CPUSIM_T c) { return __cpusim_sat(__hfma(a, b, c)); }
__CPUSIM_HD __CPUSIM_T __hfma_relu(__CPUSIM_T a, __CPUSIM_T b, __CPUSIM_T c) {
    __CPUSIM_T r = __hfma(a, b, c);
    return __cpusim_fp::is_nan(r) ? __CPUSIM_CANON_NAN : __CPUSIM_F(r) < 0.0f ? __CPUSIM_MK(0) : r;
}
// max/min: a NaN loses to a number; +0 > -0. The _nan forms propagate NaN.
__CPUSIM_HD __CPUSIM_T __hmax(__CPUSIM_T a, __CPUSIM_T b) {
    bool na = __cpusim_fp::is_nan(a), nb = __cpusim_fp::is_nan(b);
    if (na || nb) return na && nb ? __CPUSIM_CANON_NAN : na ? b : a;
    float x = __CPUSIM_F(a), y = __CPUSIM_F(b);
    if (x == y) return (a.__x & 0x8000u) ? b : a;
    return x > y ? a : b;
}
__CPUSIM_HD __CPUSIM_T __hmin(__CPUSIM_T a, __CPUSIM_T b) {
    bool na = __cpusim_fp::is_nan(a), nb = __cpusim_fp::is_nan(b);
    if (na || nb) return na && nb ? __CPUSIM_CANON_NAN : na ? b : a;
    float x = __CPUSIM_F(a), y = __CPUSIM_F(b);
    if (x == y) return (a.__x & 0x8000u) ? a : b;
    return x < y ? a : b;
}
__CPUSIM_HD __CPUSIM_T __hmax_nan(__CPUSIM_T a, __CPUSIM_T b) {
    return __cpusim_fp::is_nan(a) || __cpusim_fp::is_nan(b) ? __CPUSIM_CANON_NAN : __hmax(a, b);
}
__CPUSIM_HD __CPUSIM_T __hmin_nan(__CPUSIM_T a, __CPUSIM_T b) {
    return __cpusim_fp::is_nan(a) || __cpusim_fp::is_nan(b) ? __CPUSIM_CANON_NAN : __hmin(a, b);
}

/* -------------------------------------------------------------- comparisons */
#define __CPUSIM_CMP(NAME, EXPR)                                                                         \
    __CPUSIM_HD bool NAME(__CPUSIM_T a, __CPUSIM_T b) {                                                   \
        float x = __CPUSIM_F(a), y = __CPUSIM_F(b);                                                       \
        return EXPR;                                                                                      \
    }
__CPUSIM_CMP(__heq, x == y)
__CPUSIM_CMP(__hne, x < y || x > y)
__CPUSIM_CMP(__hlt, x < y)
__CPUSIM_CMP(__hle, x <= y)
__CPUSIM_CMP(__hgt, x > y)
__CPUSIM_CMP(__hge, x >= y)
__CPUSIM_CMP(__hequ, !(x < y || x > y))
__CPUSIM_CMP(__hneu, !(x == y))
__CPUSIM_CMP(__hltu, !(x >= y))
__CPUSIM_CMP(__hleu, !(x > y))
__CPUSIM_CMP(__hgtu, !(x <= y))
__CPUSIM_CMP(__hgeu, !(x < y))
#undef __CPUSIM_CMP
__CPUSIM_HD bool __hisnan(__CPUSIM_T a) { return __cpusim_fp::is_nan(a); }
__CPUSIM_HD int __hisinf(__CPUSIM_T a) {
    float f = __CPUSIM_F(a);
    return f == __builtin_inff() ? 1 : f == -__builtin_inff() ? -1 : 0;
}

/* --------------------------------------------------------------------- math */
#define __CPUSIM_MATH(STEM, EXPR)                                                              \
    __CPUSIM_HD __CPUSIM_T h##STEM(__CPUSIM_T a) {                                              \
        float x = __CPUSIM_F(a);                                                                \
        return __CPUSIM_RN(EXPR);                                                               \
    }                                                                                           \
    __CPUSIM_HD __CPUSIM_T2 h2##STEM(__CPUSIM_T2 a) { return __CPUSIM_T2(h##STEM(a.x), h##STEM(a.y)); }
__CPUSIM_MATH(sqrt, __builtin_sqrtf(x))
__CPUSIM_MATH(rsqrt, 1.0f / __builtin_sqrtf(x))
__CPUSIM_MATH(rcp, 1.0f / x)
__CPUSIM_MATH(exp, __builtin_expf(x))
__CPUSIM_MATH(exp2, __builtin_exp2f(x))
__CPUSIM_MATH(exp10, __builtin_expf(x * 2.302585092994046f))
__CPUSIM_MATH(log, __builtin_logf(x))
__CPUSIM_MATH(log2, __builtin_log2f(x))
__CPUSIM_MATH(log10, __builtin_log10f(x))
__CPUSIM_MATH(sin, __builtin_sinf(x))
__CPUSIM_MATH(cos, __builtin_cosf(x))
__CPUSIM_MATH(tanh, __builtin_tanhf(x))
__CPUSIM_MATH(ceil, __builtin_ceilf(x))
__CPUSIM_MATH(floor, __builtin_floorf(x))
__CPUSIM_MATH(trunc, __builtin_truncf(x))
__CPUSIM_MATH(rint, __builtin_rintf(x))
#undef __CPUSIM_MATH

/* -------------------------------------------------------- pair arithmetic */
#define __CPUSIM_PAIR2(NAME, OP)                                                                       \
    __CPUSIM_HD __CPUSIM_T2 NAME(__CPUSIM_T2 a, __CPUSIM_T2 b) { return __CPUSIM_T2(OP(a.x, b.x), OP(a.y, b.y)); }
__CPUSIM_PAIR2(__hadd2, __hadd)
__CPUSIM_PAIR2(__hsub2, __hsub)
__CPUSIM_PAIR2(__hmul2, __hmul)
__CPUSIM_PAIR2(__h2div, __hdiv)
__CPUSIM_PAIR2(__hadd2_rn, __hadd)
__CPUSIM_PAIR2(__hsub2_rn, __hsub)
__CPUSIM_PAIR2(__hmul2_rn, __hmul)
__CPUSIM_PAIR2(__hadd2_sat, __hadd_sat)
__CPUSIM_PAIR2(__hsub2_sat, __hsub_sat)
__CPUSIM_PAIR2(__hmul2_sat, __hmul_sat)
__CPUSIM_PAIR2(__hmax2, __hmax)
__CPUSIM_PAIR2(__hmin2, __hmin)
__CPUSIM_PAIR2(__hmax2_nan, __hmax_nan)
__CPUSIM_PAIR2(__hmin2_nan, __hmin_nan)
#undef __CPUSIM_PAIR2
__CPUSIM_HD __CPUSIM_T2 __hfma2(__CPUSIM_T2 a, __CPUSIM_T2 b, __CPUSIM_T2 c) {
    return __CPUSIM_T2(__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y));
}
__CPUSIM_HD __CPUSIM_T2 __hfma2_sat(__CPUSIM_T2 a, __CPUSIM_T2 b, __CPUSIM_T2 c) {
    return __CPUSIM_T2(__hfma_sat(a.x, b.x, c.x), __hfma_sat(a.y, b.y, c.y));
}
__CPUSIM_HD __CPUSIM_T2 __hfma2_relu(__CPUSIM_T2 a, __CPUSIM_T2 b, __CPUSIM_T2 c) {
    return __CPUSIM_T2(__hfma_relu(a.x, b.x, c.x), __hfma_relu(a.y, b.y, c.y));
}
__CPUSIM_HD __CPUSIM_T2 __hneg2(__CPUSIM_T2 a) { return __CPUSIM_T2(__hneg(a.x), __hneg(a.y)); }
__CPUSIM_HD __CPUSIM_T2 __habs2(__CPUSIM_T2 a) { return __CPUSIM_T2(__habs(a.x), __habs(a.y)); }
__CPUSIM_HD __CPUSIM_T2 __hisnan2(__CPUSIM_T2 a) {
    return __CPUSIM_T2(__CPUSIM_RN(__hisnan(a.x) ? 1.0f : 0.0f), __CPUSIM_RN(__hisnan(a.y) ? 1.0f : 0.0f));
}
// Comparisons of pairs: 1.0/0.0 per half, a 0xffff/0 mask per half, or both true.
#define __CPUSIM_CMP2(STEM)                                                                                       \
    __CPUSIM_HD __CPUSIM_T2 __CPUSIM_NAME3(__h, STEM, 2)(__CPUSIM_T2 a, __CPUSIM_T2 b) {                          \
        return __CPUSIM_T2(__CPUSIM_RN(__CPUSIM_NAME(__h, STEM)(a.x, b.x) ? 1.0f : 0.0f),                         \
                           __CPUSIM_RN(__CPUSIM_NAME(__h, STEM)(a.y, b.y) ? 1.0f : 0.0f));                        \
    }                                                                                                             \
    __CPUSIM_HD unsigned __CPUSIM_NAME3(__h, STEM, 2_mask)(__CPUSIM_T2 a, __CPUSIM_T2 b) {                        \
        return (__CPUSIM_NAME(__h, STEM)(a.x, b.x) ? 0xffffu : 0u) | (__CPUSIM_NAME(__h, STEM)(a.y, b.y) ? 0xffff0000u : 0u); \
    }                                                                                                             \
    __CPUSIM_HD bool __CPUSIM_NAME3(__hb, STEM, 2)(__CPUSIM_T2 a, __CPUSIM_T2 b) {                                \
        return __CPUSIM_NAME(__h, STEM)(a.x, b.x) && __CPUSIM_NAME(__h, STEM)(a.y, b.y);                          \
    }
__CPUSIM_CMP2(eq)
__CPUSIM_CMP2(ne)
__CPUSIM_CMP2(lt)
__CPUSIM_CMP2(le)
__CPUSIM_CMP2(gt)
__CPUSIM_CMP2(ge)
__CPUSIM_CMP2(equ)
__CPUSIM_CMP2(neu)
__CPUSIM_CMP2(ltu)
__CPUSIM_CMP2(leu)
__CPUSIM_CMP2(gtu)
__CPUSIM_CMP2(geu)
#undef __CPUSIM_CMP2

/* ---------------------------------------------------------------- operators */
__CPUSIM_HD __CPUSIM_T operator+(const __CPUSIM_T& a, const __CPUSIM_T& b) { return __hadd(a, b); }
__CPUSIM_HD __CPUSIM_T operator-(const __CPUSIM_T& a, const __CPUSIM_T& b) { return __hsub(a, b); }
__CPUSIM_HD __CPUSIM_T operator*(const __CPUSIM_T& a, const __CPUSIM_T& b) { return __hmul(a, b); }
__CPUSIM_HD __CPUSIM_T operator/(const __CPUSIM_T& a, const __CPUSIM_T& b) { return __hdiv(a, b); }
__CPUSIM_HD __CPUSIM_T& operator+=(__CPUSIM_T& a, const __CPUSIM_T& b) { return a = __hadd(a, b); }
__CPUSIM_HD __CPUSIM_T& operator-=(__CPUSIM_T& a, const __CPUSIM_T& b) { return a = __hsub(a, b); }
__CPUSIM_HD __CPUSIM_T& operator*=(__CPUSIM_T& a, const __CPUSIM_T& b) { return a = __hmul(a, b); }
__CPUSIM_HD __CPUSIM_T& operator/=(__CPUSIM_T& a, const __CPUSIM_T& b) { return a = __hdiv(a, b); }
__CPUSIM_HD __CPUSIM_T operator+(const __CPUSIM_T& a) { return a; }
__CPUSIM_HD __CPUSIM_T operator-(const __CPUSIM_T& a) { return __hneg(a); }
__CPUSIM_HD __CPUSIM_T& operator++(__CPUSIM_T& a) { return a = __hadd(a, __CPUSIM_RN(1.0f)); }
__CPUSIM_HD __CPUSIM_T& operator--(__CPUSIM_T& a) { return a = __hsub(a, __CPUSIM_RN(1.0f)); }
__CPUSIM_HD __CPUSIM_T operator++(__CPUSIM_T& a, int) { __CPUSIM_T r = a; ++a; return r; }
__CPUSIM_HD __CPUSIM_T operator--(__CPUSIM_T& a, int) { __CPUSIM_T r = a; --a; return r; }
__CPUSIM_HD bool operator==(const __CPUSIM_T& a, const __CPUSIM_T& b) { return __heq(a, b); }
__CPUSIM_HD bool operator!=(const __CPUSIM_T& a, const __CPUSIM_T& b) { return __hneu(a, b); }
__CPUSIM_HD bool operator<(const __CPUSIM_T& a, const __CPUSIM_T& b) { return __hlt(a, b); }
__CPUSIM_HD bool operator<=(const __CPUSIM_T& a, const __CPUSIM_T& b) { return __hle(a, b); }
__CPUSIM_HD bool operator>(const __CPUSIM_T& a, const __CPUSIM_T& b) { return __hgt(a, b); }
__CPUSIM_HD bool operator>=(const __CPUSIM_T& a, const __CPUSIM_T& b) { return __hge(a, b); }

__CPUSIM_HD __CPUSIM_T2 operator+(const __CPUSIM_T2& a, const __CPUSIM_T2& b) { return __hadd2(a, b); }
__CPUSIM_HD __CPUSIM_T2 operator-(const __CPUSIM_T2& a, const __CPUSIM_T2& b) { return __hsub2(a, b); }
__CPUSIM_HD __CPUSIM_T2 operator*(const __CPUSIM_T2& a, const __CPUSIM_T2& b) { return __hmul2(a, b); }
__CPUSIM_HD __CPUSIM_T2 operator/(const __CPUSIM_T2& a, const __CPUSIM_T2& b) { return __h2div(a, b); }
__CPUSIM_HD __CPUSIM_T2& operator+=(__CPUSIM_T2& a, const __CPUSIM_T2& b) { return a = __hadd2(a, b); }
__CPUSIM_HD __CPUSIM_T2& operator-=(__CPUSIM_T2& a, const __CPUSIM_T2& b) { return a = __hsub2(a, b); }
__CPUSIM_HD __CPUSIM_T2& operator*=(__CPUSIM_T2& a, const __CPUSIM_T2& b) { return a = __hmul2(a, b); }
__CPUSIM_HD __CPUSIM_T2& operator/=(__CPUSIM_T2& a, const __CPUSIM_T2& b) { return a = __h2div(a, b); }
__CPUSIM_HD __CPUSIM_T2 operator+(const __CPUSIM_T2& a) { return a; }
__CPUSIM_HD __CPUSIM_T2 operator-(const __CPUSIM_T2& a) { return __hneg2(a); }
__CPUSIM_HD bool operator==(const __CPUSIM_T2& a, const __CPUSIM_T2& b) { return __hbeq2(a, b); }
__CPUSIM_HD bool operator!=(const __CPUSIM_T2& a, const __CPUSIM_T2& b) { return __hbneu2(a, b); }
__CPUSIM_HD bool operator<(const __CPUSIM_T2& a, const __CPUSIM_T2& b) { return __hblt2(a, b); }
__CPUSIM_HD bool operator<=(const __CPUSIM_T2& a, const __CPUSIM_T2& b) { return __hble2(a, b); }
__CPUSIM_HD bool operator>(const __CPUSIM_T2& a, const __CPUSIM_T2& b) { return __hbgt2(a, b); }
__CPUSIM_HD bool operator>=(const __CPUSIM_T2& a, const __CPUSIM_T2& b) { return __hbge2(a, b); }

#undef __CPUSIM_MK
#undef __CPUSIM_RN
#undef __CPUSIM_F
#undef __CPUSIM_CANON_NAN
#undef __CPUSIM_NAME4
#undef __CPUSIM_NAME3
#undef __CPUSIM_NAME
#undef __CPUSIM_CAT
#undef __CPUSIM_CAT_
