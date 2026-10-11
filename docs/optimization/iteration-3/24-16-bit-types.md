# 24. 16-bit floating-point types

> **In one sentence:** the `cuda_fp16.h` and `cuda_bf16.h` headers now exist for `cucc`, with the CUDA API for `__half`, `__half2`, `__nv_bfloat16` and `__nv_bfloat162`; since Zen 4 has no 16-bit arithmetic, each operation is done in `float` and rounded once to the format, which gives the same correctly rounded result as the GPU by the double rounding theorem, and a test checks it bit by bit against an independent model.

The general concepts (lanes, per-thread code, variants) are in [00-basic-concepts.md](../00-basic-concepts.md). The two formats:

| Format | Sign | Exponent | Fraction | Precision `p` (with the implicit bit) | Largest finite |
|---|---|---|---|---|---|
| half (IEEE binary16) | 1 | 5 | 10 | 11 | 65504 |
| bfloat16 | 1 | 8 | 7 | 8 | ~3.39e38 |
| float (binary32) | 1 | 8 | 23 | 24 | ~3.40e38 |

bfloat16 is literally the upper half of the bits of a `float`: same exponent, fraction truncated to 7 bits.

## The problem

CUDA programs with `#include <cuda_fp16.h>` or `<cuda_bf16.h>` did not compile: the headers did not exist. These types appear in machine learning code, in kernels that store data in half precision to reduce bandwidth, and they are the input of the tensor cores ([25](25-tensor-cores.md)).

Implementing them is not just declaring a 16-bit `struct`. On the GPU, `__hadd`, `__hmul`, `__hdiv`, `hsqrt` and `__hfma` are **correctly rounded**: the result is the exact value of the operation rounded once to the format. An implementation that differed by 1 ulp in some cases would break the bit-by-bit comparison with the GPU that the project uses for validation ([23](23-reproducibility-and-stability.md)).

The available hardware:

- Zen 4 has **F16C**: `vcvtph2ps` (half to float, exact) and `vcvtps2ph` (float to half, with the rounding mode in an immediate), up to 16 elements per instruction with AVX-512.
- Zen 4 **does not have AVX512-FP16**, the extension with native half arithmetic (`vaddph`, `vfmadd...ph`). There is no 16-bit add or multiply.
- For bfloat16, Zen 4 has AVX512-BF16, but it only provides conversions and the dot product `vdpbf16ps` (used by the tensor cores, see [25](25-tensor-cores.md)); there is no element-wise arithmetic either.

## The idea

Do each operation in `float` and round once to the 16-bit format. The question is whether this is correct: the `float` result is already rounded (to 24 bits), and then it is rounded again (to `p` bits). Rounding twice can be wrong: if the first rounding lands exactly on the midpoint between two `p`-bit values, the second breaks the tie to even, when the exact value was on one side of the midpoint.

**The double rounding theorem** (Figueroa, 1995): for operands of `p` bits and an operation in {+, -, *, /, square root}, if `q >= 2p + 2`, then rounding the exact result to `q` bits and then to `p` bits (both to nearest) gives the same as rounding once to `p` bits. The reason: the exact result of these operations cannot fall too close to a `p`-bit midpoint without being exactly the midpoint (a product of two `p`-bit numbers has at most `2p` bits; an inexact quotient or root stays at a minimum distance from the midpoints that depends on `p`), and with `q >= 2p + 2` rounding to `q` bits cannot move the value onto the midpoint.

With `float`, `q = 24`:

- half: `p = 11`, `2p + 2 = 24 <= 24`. It holds, at the exact limit.
- bfloat16: `p = 8`, `2p + 2 = 18 <= 24`. It holds with margin.

In the subnormal range of half, `float` is still normal (the smallest subnormal of half is 2^-24), so the condition continues to hold. In the subnormal range of bfloat16, `float` is also subnormal, with the same minimum exponent, but with 16 more bits of fraction: if bfloat16 has `p' <= 7` effective bits, `float` has `p' + 16 >= 2p' + 2`.

The **FMA** is not covered by the theorem: `a*b + c` with `p`-bit operands has a sum whose exact result can have many more than `2p` bits. For it, another tool: **round to odd** (*round to odd*, Boldo and Melquiond): truncate and, if the result is inexact, force the last bit to 1. If `x` is rounded to odd to `q >= p + 2` bits and then to nearest to `p` bits, the result equals rounding `x` to nearest directly. The forced bit works as a *sticky bit*: it prevents an inexact value from looking like exactly a midpoint. `__hfma` computes the exact product in `double` (22 bits in half, 16 in bfloat16, which fit in the 53 of `double`), adds `c` rounding to odd in `double`, converts to `float` again rounding to odd, and only then rounds to nearest to 16 bits. The conversions from `double` to 16 bits use the same path.

The **directed modes** (`_rz`, `_rd`, `_ru`) are derived from the round-to-nearest result: it is within 1 ulp of the exact value, so either it is already the right one or the right one is its neighbor. Since both formats are sign-magnitude, the neighbor is the bit pattern plus or minus 1 (with the direction given by the sign), including across the boundary between exponents, because the exponent sits above the fraction and the increment propagates into it.

## How it works, step by step

### Conversions

- **half to float:** `vcvtph2ps`, exact.
- **float to half, to nearest:** `vcvtps2ph`. In C++ code, `(_Float16)f` is enough; Clang generates the instruction.
- **bfloat16 to float:** shift 16 bits to the left; exact.
- **float to bfloat16, to nearest even:** addition on the bits. Example with the 32 bits `u` of a `float`:

```text
u          = 0x3FC0_8000   (1.50390625: exactly between 0x3FC0 and 0x3FC1)
lsb        = (u >> 16) & 1 = 0
u + 0x7FFF + lsb = 0x3FC0_FFFF  -> >> 16 = 0x3FC0   (tie: stays at the even one)

u          = 0x3FC1_8000   (exactly between 0x3FC1 and 0x3FC2)
lsb        = 1
u + 0x7FFF + 1   = 0x3FC2_0000  -> >> 16 = 0x3FC2   (tie: goes up to the even one)
```

Below the midpoint the addition does not reach bit 16; above it, it does. Overflow also comes out right: the largest `float` rounds to `0x7F80`, the infinity of bfloat16. NaN needs a separate case, because a NaN whose payload is only in the low 16 bits would become infinity on truncation.

### Operations

`__hadd(a, b)` is `RN16(float(a) + float(b))`; the same for `-`, `*`, `/` and `hsqrt`. The math functions (`hexp`, `hsin`...) follow the same scheme; for them the GPU does not promise correct rounding, and the result here is that of `expf`, `sinf`... rounded.

In the vector code generated by the SIMT compiler, each lane does these computations in `float`, so a `__half` addition becomes `vcvtph2ps`, `vaddps` and `vcvtps2ph` on 16 lanes per instruction.

### Conversions to integer

Saturating, as in CUDA: the value is rounded in the requested mode, NaN gives 0, and values outside the range of the type give the minimum or the maximum. The implicit C++ conversion (`(int)h`) uses truncation toward zero, like the CUDA one.

### The API

The API has the same names as CUDA, including the implicit conversions: `__half` has an implicit constructor from `float` and an implicit conversion operator to `float`. This reproduces the same ambiguities as CUDA: `h * 2.0f` is ambiguous (both `operator*(__half, __half)`, converting `2.0f`, and the native `*` of `float`, converting `h`, require a user-defined conversion), and the same code is rejected by `nvcc` and by `cucc`.

The code is written once, in `__cpusim_fp16_impl.h`, and included twice with macros: one for `__half` and one for `__nv_bfloat16`. The differences between the formats are in four macros (conversion to `float`, conversion from `float` with mode, infinity bits and name stems).

## The code

### The basic conversions

`include/__cpusim_fp16_common.h:33-43`

```cpp
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
```

- `half_rn` and `half_tof` use Clang's `_Float16` type; its conversions become the F16C instructions.
- In `bf16_rn`, `| 0x40u` sets the highest bit of the bfloat16 fraction (bit 6 of the 16), which is the quiet NaN bit: the result is always a NaN, never infinity.
- The last line is the addition from the example above.

### The directed modes

`include/__cpusim_fp16_common.h:47-63`

```cpp
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
```

`back` is the value of `r` converted back to `float` (exact). For `RZ`: if `|back| > |f|`, round-to-nearest rounded away from zero, and `r - 1` decreases the magnitude. For `RD`: if `back > f`, it must go down; for a negative number, going down means increasing the magnitude (`r + 1`), for a positive one it means decreasing it (`r - 1`). The edge cases come out without special handling:

- overflow with `RZ`: `r` is infinity (`0x7C00` in half), `back > f`, and `r - 1 = 0x7BFF` is the largest finite value;
- `-tiny` with `RD`: the nearest is `-0` (`0x8000`), `back = -0 > f`, and `r + 1 = 0x8001` is the smallest negative subnormal.

### Round to odd

`include/__cpusim_fp16_common.h:65-72`

```cpp
// double -> float rounded to odd: truncate, then set the last bit if inexact.
__CPUSIM_HD float d2f_odd(double d) {
    float f = (float)d;
    if (d != d) return f;
    if (__builtin_fabs((double)f) > __builtin_fabs(d)) f = u2f(f2u(f) - 1);
    if ((double)f != d) f = u2f(f2u(f) | 1u);
    return f;
}
```

`(float)d` rounds to nearest; if that increased the magnitude, the `-1` step on the bits goes back to truncation (same sign-magnitude argument as the directed modes). If the truncated result is not equal to `d`, the last bit is forced to 1.

### The arithmetic

`include/__cpusim_fp16_impl.h:153-167`

```cpp
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
    double r = __cpusim_fp::add_odd((double)__CPUSIM_F(a) * (double)__CPUSIM_F(b), (double)__CPUSIM_F(c));
    return __CPUSIM_RN(__cpusim_fp::d2f_odd(r));
}
```

The `double` step uses the sum with exact error (TwoSum, Knuth): `s = x + y` rounded and `e` such that `x + y = s + e` exactly. If `e` is not zero and the last bit of `s` is even, `s` moves one unit in the direction of `e`, which gives the round-to-odd of the exact sum in 53 bits.

`include/__cpusim_fp16_common.h:74-86`

```cpp
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
```

`__CPUSIM_F` converts to `float` and `__CPUSIM_RN` rounds to nearest to the format. The CUDA `_rn` forms exist to prevent `nvcc` from contracting `__hmul` + `__hadd` into an FMA; here each operation already rounds to 16 bits at the end, and the compiler cannot fuse across that conversion.

### The instantiation for the two types

`include/cuda_fp16.h:13-22`

```cpp
#define __CPUSIM_T __half
#define __CPUSIM_T2 __half2
#define __CPUSIM_RAW __half_raw
#define __CPUSIM_RAW2 __half2_raw
#define __CPUSIM_N half
#define __CPUSIM_N2 half2
#define __CPUSIM_TOF(bits) __cpusim_fp::half_tof(bits)
#define __CPUSIM_FROM(f, mode) __cpusim_fp::half_from((f), (mode))
#define __CPUSIM_INF_BITS 0x7c00u
#include "__cpusim_fp16_impl.h"
```

`cuda_bf16.h` does the same with `__nv_bfloat16`, `bfloat16`, `bf16_tof`, `bf16_from` and `0x7f80u`. `__CPUSIM_N` assembles the names: `__float2half_rz` and `__float2bfloat16_rz` come from the same line.

### The implicit conversions

`include/__cpusim_fp16_impl.h:32-46`

```cpp
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
```

One constructor per integer type prevents an `int` from choosing between `float` and `double` by conversion (which would be ambiguous). From `double` and from integers the path is round to odd.

### What went into the device header

`include/__cpusim_device.h` gained other small-type intrinsics that were missing: `__dp4a` and `__dp2a_lo/hi` (8-bit and 16-bit dot products inside a 32-bit word), the SIMD-within-a-word operations (`__vadd4`, `__vsadu4`, `__vmaxu2`...) and the loads with cache hints (`__ldg`, `__ldcg`, `__ldcs`...), of which not even `__ldg` existed before.

`include/__cpusim_device.h:321-333`

```cpp
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
```

They are written in plain per-thread C++: the loop has a fixed count and is unrolled, and the SIMT compiler turns them into vector operations over the lanes like any other integer computation.

`include/__cpusim_device.h:388-390`

```cpp
/* Cached loads and stores: the cache hints mean nothing on the CPU. */
template <typename T>
__CPUSIM_DEV T __ldg(const T* p) { return *p; }
```

On the GPU, `__ldg` reads through the (read-only) texture cache and the variants `__ldcg`, `__ldcs`... choose the cache policy. On the CPU there is no useful equivalent; the ordinary load goes through the same access analysis ([12](../iteration-2/12-static-access-analysis.md)) as any other.

### The model in the test

`tests/cuda/types.cu:20-36`

```cpp
// All finite values of a 16-bit format in increasing order, with their bits.
struct Format {
    std::vector<double> value;
    std::vector<uint16_t> bits;
    uint16_t inf, nan_threshold;
    explicit Format(bool bf16) {
        inf = bf16 ? 0x7f80 : 0x7c00;
        nan_threshold = inf;
        std::vector<std::pair<double, uint16_t>> all;
        for (unsigned b = 0; b < 0x10000; ++b) {
            if ((b & 0x7fff) >= inf) continue;  // inf and NaN
            if (b == 0x8000) continue;          // -0: same value as +0
            all.push_back({decode(uint16_t(b), bf16), uint16_t(b)});
        }
        std::sort(all.begin(), all.end());
        for (auto& [v, b] : all) value.push_back(v), bits.push_back(b);
    }
```

The model shares no code with the headers: it walks the 65536 bit patterns, keeps the finite values sorted, and rounds any `double` by binary search in that table (`std::upper_bound`), choosing the lower or upper neighbor according to the mode. The operations are done in `double`, where the sum, difference and product of two 16-bit values are exact, and division and square root are rounded twice with margin (`53 >= 2 * 11 + 2`).

## When it applies (and when it does not)

- Any code that includes `cuda_fp16.h` or `cuda_bf16.h`, in kernels and on the host (the headers also work in plain C++, without `__CUDA__`).
- The math functions (`hexp`, `hlog`, `hsin`...) are not correctly rounded, not even on the GPU; the result can differ from the GPU's by 1 ulp.
- Performance: each operation costs the conversion of the operands, the computation in `float` and the conversion back, versus one instruction on a GPU with native half arithmetic (which with `__half2` still does two operations per instruction). The memory bandwidth gain (half the bytes of a `float`) exists equally on the CPU.

## Results

- `tests/cuda/types.cu` checks bit by bit about 24 operations per input in each format (arithmetic, FMA, square root, maximum and minimum, comparisons, conversions from `float`, `double` and `int` in the four modes, conversions to integer, pair operations, shuffles), plus `int8`, `int16`, `int32`, `int64`, `fp32`, `fp64` and small types passed as kernel arguments. It passes in all 12 execution modes.
- The first failure of the test was in the test's own model (infinite input with directed rounding), not in the headers.
- **Compile time:** the test compiles in 13.6 s, almost all of it in the register allocation of `cpusim-devc` (`perf` shows `LiveIntervalUnion` and `SpillPlacement` at the top). A single kernel gathers the whole API and is generated at each width; see the variant budget in [26](26-compiler-and-runtime-changes.md).

## Limitations and trade-offs

- **FMA in bfloat16 with `c` much smaller than `a*b` (fixed):** the first version added `a*b + c` with a plain `double` FMA. In bfloat16, with an 8-bit exponent, `c` can fall below half an ulp of `a*b` in `double` and disappear; if `a*b` is exactly an 8-bit midpoint, the tie was broken to even instead of by the sign of `c`. Example: `a = 1 + 2^-7`, `b = 1.5`, `c = -2^-100`; `a*b = 1.51171875` is the midpoint between 1.5078125 and 1.515625 and the correct result is 1.5078125, but 1.515625 came out. The sum with round to odd (`add_odd`, above) fixed it; the `fma_ties` kernel of `tests/cuda/types.cu` tests this case and the equivalent in half.
- **64-bit integers to bfloat16:** the conversion goes through `double`, which already rounds values above 2^53; for those values the sequence can hit the same midpoint problem. The comment in the code records the guarantee only for 32-bit values.
- **Compile time** of kernels that use many API functions at the same time (see Results).

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): lanes and AVX-512.
- [23-reproducibility-and-stability.md](23-reproducibility-and-stability.md): the execution modes in which the test runs.
- [25-tensor-cores.md](25-tensor-cores.md): the tensor cores, which take these types as input.
- [26-compiler-and-runtime-changes.md](26-compiler-and-runtime-changes.md): the size budget of the generic variants.
