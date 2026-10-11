// Numeric types in device code: int8, int16, int32, int64 (signed and
// unsigned), fp32, fp64, half (__half/__half2) and bfloat16
// (__nv_bfloat16/__nv_bfloat162).
//
// The 16-bit float results are checked bit for bit against a host model that
// does not share code with cuda_fp16.h/cuda_bf16.h: every finite value of the
// format sorted in a table, rounding by binary search in it (nearest even,
// toward zero, down, up), and the operations evaluated in double, where they
// are exact or (division, square root) safely double-rounded.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <vector>

#include "check.cuh"

/* ------------------------------------------------------------ host model */

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
    static double decode(uint16_t b, bool bf16) {
        if (bf16) {
            uint32_t u = uint32_t(b) << 16;
            float f;
            std::memcpy(&f, &u, 4);
            return f;
        }
        int e = (b >> 10) & 31, m = b & 1023;
        double v = e == 0 ? std::ldexp(m, -24) : e == 31 ? INFINITY : std::ldexp(1024 + m, e - 25);
        return (b & 0x8000) ? -v : v;
    }
    // Rounds x (any double) to the format: mode 0 nearest even, 1 toward zero, 2 down, 3 up.
    uint16_t round(double x, int mode) const {
        if (std::isnan(x)) return 0x7fff;
        const uint16_t sign = std::signbit(x) ? 0x8000 : 0;
        if (x == 0) return sign;
        if (std::isinf(x)) return uint16_t(inf | sign);
        const double max = value.back();
        // Overflow follows IEEE: nearest goes to inf at max + half an ulp.
        auto lo_it = std::upper_bound(value.begin(), value.end(), x);
        uint16_t r;
        if (x > max || x < -max) {
            bool pos = x > 0;
            double ulp = max - value[value.size() - 2];
            bool to_inf = mode == 0 ? std::fabs(x) >= max + ulp / 2 : mode == 1 ? false : mode == 2 ? !pos : pos;
            if (to_inf) return uint16_t(inf | (pos ? 0 : 0x8000));
            return pos ? bits.back() : bits.front();
        }
        size_t hi = size_t(lo_it - value.begin()), lo = hi - 1;  // value[lo] <= x < value[hi]
        if (value[lo] == x) {
            r = bits[lo];
        } else {
            double a = value[lo], b = value[hi];
            size_t pick;
            if (mode == 0) {
                double da = x - a, db = b - x;
                pick = da < db ? lo : db < da ? hi : (bits[lo] & 1) == 0 ? lo : hi;
            } else if (mode == 1) {
                pick = x > 0 ? lo : hi;
            } else {
                pick = mode == 2 ? lo : hi;
            }
            r = bits[pick];
        }
        // A zero result keeps the sign of x (round toward zero of -tiny is -0).
        if ((r & 0x7fff) == 0) r = sign;
        return r;
    }
    double to_double(uint16_t b) const {
        if ((b & 0x7fff) > inf) return NAN;
        if ((b & 0x7fff) == inf) return (b & 0x8000) ? -INFINITY : INFINITY;
        return decode(b, inf == 0x7f80);
    }
    bool is_nan(uint16_t b) const { return (b & 0x7fff) > inf; }
};

/* ---------------------------------------------------------------- kernels */

constexpr int kOps = 24;  // results per input for each 16-bit type

// Ops on (a, b, c) of a 16-bit type T; out[k] are raw bits or 0/1 flags.
template <typename T, typename T2>
__device__ void ops16(T a, T b, T c, float f, double d, int iv, unsigned short* out) {
    out[0] = __hadd(a, b).__x;
    out[1] = __hsub(a, b).__x;
    out[2] = __hmul(a, b).__x;
    out[3] = __hdiv(a, b).__x;
    out[4] = __hfma(a, b, c).__x;
    out[5] = hsqrt(__habs(a)).__x;
    out[6] = (-a).__x;
    out[7] = __hmax(a, b).__x;
    out[8] = __hmin(a, b).__x;
    out[9] = (a < b) | (a == b) << 1 | __hneu(a, b) << 2 | __hgeu(a, b) << 3 | __hisnan(a) << 4;
    out[10] = T(f).__x;  // float -> T, nearest even
    out[11] = T(d).__x;  // double -> T
    out[12] = T(iv).__x;  // int -> T
    out[13] = (unsigned short)(int)a;  // T -> int, toward zero (low 16 bits)
    out[14] = (unsigned short)(short)a;  // saturating short
    T2 p = T2(a, b), q = T2(c, a);
    T2 s = __hfma2(p, q, p);
    out[15] = s.x.__x;
    out[16] = s.y.__x;
    T acc = a;
    acc += b;
    acc *= c;
    out[17] = acc.__x;
    out[18] = __shfl_xor_sync(0xffffffffu, a, 3).__x;
    out[19] = __shfl_down_sync(0xffffffffu, p, 1).y.__x;
    out[20] = __hmul_sat(a, b).__x;
    out[21] = (unsigned short)(__hbgt2(p, q) | __hblt2(p, q) << 1);
    out[22] = __float2half_rz(f).__x;  // replaced per type below
    out[23] = 0;
}

__global__ void half_ops(const unsigned short* in, const float* fin, const double* din, const int* iin, unsigned short* out,
                         int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __half a = __ushort_as_half(in[3 * i]), b = __ushort_as_half(in[3 * i + 1]), c = __ushort_as_half(in[3 * i + 2]);
    unsigned short* o = out + size_t(i) * kOps;
    ops16<__half, __half2>(a, b, c, fin[i], din[i], iin[i], o);
    o[22] = __float2half_rz(fin[i]).__x;
    o[23] = __float2half_rd(fin[i]).__x ^ (__float2half_ru(fin[i]).__x << 1);
}

__global__ void bf16_ops(const unsigned short* in, const float* fin, const double* din, const int* iin, unsigned short* out,
                         int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __nv_bfloat16 a = __ushort_as_bfloat16(in[3 * i]), b = __ushort_as_bfloat16(in[3 * i + 1]),
                  c = __ushort_as_bfloat16(in[3 * i + 2]);
    unsigned short* o = out + size_t(i) * kOps;
    ops16<__nv_bfloat16, __nv_bfloat162>(a, b, c, fin[i], din[i], iin[i], o);
    o[22] = __float2bfloat16_rz(fin[i]).__x;
    o[23] = __float2bfloat16_rd(fin[i]).__x ^ (__float2bfloat16_ru(fin[i]).__x << 1);
}

// A block sum of halves in shared memory (fixed order), and a gather.
__global__ void half_reduce(const __half* x, const int* idx, __half* sums, __half* gathered) {
    __shared__ __half s[128];
    int t = threadIdx.x, i = blockIdx.x * 128 + t;
    s[t] = __ldg(x + i);
    gathered[i] = x[idx[i]];
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (t < stride) s[t] = s[t] + s[t + stride];
        __syncthreads();
    }
    if (t == 0) sums[blockIdx.x] = s[0];
}

// Small integers: wrapping arithmetic, promotions, gathers, division,
// char4 loads, and the dot product / SIMD-in-word intrinsics.
struct IntOut {
    signed char s8[4];
    unsigned char u8[4];
    short s16[4];
    unsigned short u16[4];
    int dp[6];
    unsigned simd[6];
};

__global__ void small_ints(const signed char* a8, const unsigned char* b8, const short* a16, const unsigned short* b16,
                           const char4* c4, const int* idx, IntOut* out, signed char k8, short k16, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j = idx[i];
    signed char x = a8[i], y = a8[j];
    unsigned char u = b8[i], v = b8[j];
    short p = a16[i], q = a16[j];
    unsigned short r = b16[i], s = b16[j];
    IntOut o;
    o.s8[0] = (signed char)(x * y + k8);
    o.s8[1] = (signed char)(y != 0 ? x / y : x);
    o.s8[2] = (signed char)(x >> 2) ^ (signed char)(y << 3);
    o.s8[3] = x < y ? x : y;
    o.u8[0] = (unsigned char)(u * v + 7);
    o.u8[1] = (unsigned char)(v ? u % v : u);
    o.u8[2] = (unsigned char)(u - v);
    o.u8[3] = (unsigned char)((u + v) >> 1);
    o.s16[0] = (short)(p * q + k16);
    o.s16[1] = (short)(q != 0 ? p / q : p);
    o.s16[2] = (short)((p << 3) ^ (q >> 1));
    o.s16[3] = p > q ? p : q;
    o.u16[0] = (unsigned short)(r * s + 1);
    o.u16[1] = (unsigned short)(s ? r % s : r);
    o.u16[2] = (unsigned short)(r - s);
    o.u16[3] = (unsigned short)(r >> 5);
    char4 w = c4[i], z = c4[j];
    int wa = __builtin_bit_cast(int, w), za = __builtin_bit_cast(int, z);
    o.dp[0] = __dp4a(w, z, i);
    o.dp[1] = (int)__dp4a((unsigned)wa, (unsigned)za, 5u);
    o.dp[2] = __dp2a_lo(wa, za, -3);
    o.dp[3] = __dp2a_hi(wa, za, 9);
    o.dp[4] = w.x * z.w + w.y * z.z;
    o.dp[5] = (int)__vsadu4((unsigned)wa, (unsigned)za);
    o.simd[0] = __vadd4((unsigned)wa, (unsigned)za);
    o.simd[1] = __vsub2((unsigned)wa, (unsigned)za);
    o.simd[2] = __vabsdiffu4((unsigned)wa, (unsigned)za);
    o.simd[3] = __vmaxs4((unsigned)wa, (unsigned)za);
    o.simd[4] = __vcmpeq4((unsigned)wa, (unsigned)(wa ^ 0x00ff0000));
    o.simd[5] = __vavgu2((unsigned)wa, (unsigned)za);
    out[i] = o;
}

// 32 and 64-bit integers and both float widths, with conversions between them.
struct WideOut {
    int i32[3];
    unsigned u32[2];
    long long i64[4];
    unsigned long long u64[3];
    float f32[3];
    double f64[3];
};

__global__ void wide(const long long* a, const double* d, WideOut* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long x = a[i], y = a[(i * 13 + 5) % n] | 1;
    double p = d[i], q = d[(i * 7 + 1) % n];
    WideOut o;
    o.i32[0] = (int)((unsigned)x * (unsigned)y);  // wrapping, not signed overflow
    o.i32[1] = (int)(x / (int)y);
    o.i32[2] = __float2int_rn((float)p);
    o.u32[0] = (unsigned)x >> (i & 31);
    o.u32[1] = (unsigned)x % ((unsigned)y | 1u);
    o.i64[0] = (long long)((unsigned long long)x * (unsigned long long)y);
    o.i64[1] = x / y;
    o.i64[2] = x % y;
    o.i64[3] = __mul64hi(x, y);
    o.u64[0] = (unsigned long long)x * (unsigned long long)y;
    o.u64[1] = (unsigned long long)x / ((unsigned long long)y | 1ull);
    o.u64[2] = (unsigned long long)x >> (i & 63);
    o.f32[0] = (float)x;
    o.f32[1] = (float)p * (float)q + 1.0f;
    o.f32[2] = (float)(p / q);
    o.f64[0] = (double)x;
    o.f64[1] = __fma_rn(p, q, -p);
    o.f64[2] = p / q;
    out[i] = o;
}

// a*b exactly halfway between two values, c far smaller: c decides the
// rounding (a sum in plain double would drop c and round the tie to even).
__global__ void fma_ties(float* out) {
    const float tiny = 1.0f / 1073741824.0f / 1073741824.0f / 1073741824.0f / 1024.0f;  // 2^-100
    __nv_bfloat16 a = __float2bfloat16(1.0f + 1.0f / 128), b = __float2bfloat16(1.5f);
    out[0] = __bfloat162float(__hfma(a, b, __float2bfloat16(-tiny)));
    out[1] = __bfloat162float(__hfma(a, b, __float2bfloat16(tiny)));
    __half h = __float2half(1.0f + 1.0f / 1024), hb = __float2half(1.5f), sub = __ushort_as_half(0x0001);  // 2^-24
    out[2] = __half2float(__hfma(h, hb, __hneg(sub)));
    out[3] = __half2float(__hfma(h, hb, sub));
}

// Small types passed by value as kernel arguments.
__global__ void args(__half h, __nv_bfloat16 b, signed char c, short s, unsigned char u, double d, long long l, float* out) {
    out[0] = __half2float(h);
    out[1] = __bfloat162float(b);
    out[2] = c;
    out[3] = s;
    out[4] = u;
    out[5] = (float)d;
    out[6] = (float)l;
}

/* ------------------------------------------------------------------ host */

static uint32_t rng_state = 2463534242u;
static uint32_t rnd() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

// Inputs for a 16-bit format: mostly values in +-[1/64, 64] (where double
// evaluates a*b+c exactly), plus specials and the extremes of the range.
static uint16_t pick16(const Format& fm, int i) {
    if (i % 17 == 0) {
        static const uint16_t half_special[] = {0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0x0001, 0x8001, 0x03ff, 0x0400, 0x7bff, 0xfbff, 0x3c00};
        static const uint16_t bf_special[] = {0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0x0001, 0x8001, 0x007f, 0x0080, 0x7f7f, 0xff7f, 0x3f80};
        return (fm.inf == 0x7c00 ? half_special : bf_special)[(i / 17) % 12];
    }
    double v = std::ldexp(1.0 + (rnd() % 4096) / 4096.0, int(rnd() % 13) - 6) * ((rnd() & 1) ? -1 : 1);
    return fm.round(v, 0);
}

static bool same16(const Format& fm, uint16_t got, uint16_t want) {
    return got == want || (fm.is_nan(got) && fm.is_nan(want));
}

static int check16(const char* name, const Format& fm, bool is_half, Checker& check) {
    const int n = 3072;  // whole warps: no shuffle reads an exited thread
    std::vector<unsigned short> in(3 * n);
    std::vector<float> fin(n);
    std::vector<double> din(n);
    std::vector<int> iin(n);
    for (int i = 0; i < 3 * n; ++i) in[i] = pick16(fm, i);
    for (int i = 0; i < n; ++i) {
        // Floats anywhere in (and beyond) the format's range, often between two values.
        uint32_t u = rnd();
        float f = std::ldexp(1.0f + (u % 100000) / 100000.0f, int(rnd() % (is_half ? 40 : 260)) - (is_half ? 26 : 130));
        fin[i] = (u & 1) ? -f : f;
        din[i] = std::ldexp(1.0 + (rnd() % 1000003) / 1000003.0, int(rnd() % 34) - 20) * ((rnd() & 1) ? -1 : 1);
        iin[i] = int(rnd() % 200001) - 100000;
    }
    unsigned short *d_in, *d_out;
    float* d_f;
    double* d_d;
    int* d_i;
    CHECK_CUDA(cudaMalloc(&d_in, in.size() * 2));
    CHECK_CUDA(cudaMalloc(&d_out, size_t(n) * kOps * 2));
    CHECK_CUDA(cudaMalloc(&d_f, n * 4));
    CHECK_CUDA(cudaMalloc(&d_d, n * 8));
    CHECK_CUDA(cudaMalloc(&d_i, n * 4));
    CHECK_CUDA(cudaMemcpy(d_in, in.data(), in.size() * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_f, fin.data(), n * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_d, din.data(), n * 8, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_i, iin.data(), n * 4, cudaMemcpyHostToDevice));
    if (is_half)
        half_ops<<<(n + 95) / 96, 96>>>(d_in, d_f, d_d, d_i, d_out, n);
    else
        bf16_ops<<<(n + 95) / 96, 96>>>(d_in, d_f, d_d, d_i, d_out, n);
    CHECK_CUDA(cudaGetLastError());
    std::vector<unsigned short> out(size_t(n) * kOps);
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost));

    char what[64];
    auto expect16 = [&](int i, int k, uint16_t want) {
        std::snprintf(what, sizeof what, "%s op %d", name, k);
        uint16_t got = out[size_t(i) * kOps + k];
        check.expect(same16(fm, got, want), what, i, got, want);
    };
    auto flag = [&](int i, int k, unsigned want) {
        std::snprintf(what, sizeof what, "%s op %d", name, k);
        unsigned got = out[size_t(i) * kOps + k];
        check.expect(got == want, what, i, got, want);
    };
    for (int i = 0; i < n; ++i) {
        uint16_t ab = in[3 * i], bb = in[3 * i + 1], cb = in[3 * i + 2];
        double a = fm.to_double(ab), b = fm.to_double(bb), c = fm.to_double(cb);
        bool na = std::isnan(a), nb = std::isnan(b);
        expect16(i, 0, fm.round(a + b, 0));
        expect16(i, 1, fm.round(a - b, 0));
        expect16(i, 2, fm.round(a * b, 0));
        expect16(i, 3, fm.round(a / b, 0));
        if (std::fabs(a) <= 64 && std::fabs(b) <= 64 && std::fabs(c) <= 64) expect16(i, 4, fm.round(a * b + c, 0));
        expect16(i, 5, fm.round(std::sqrt(std::fabs(a)), 0));
        expect16(i, 6, uint16_t(ab ^ 0x8000));
        // max/min: NaN loses; +0 beats -0.
        uint16_t mx = na ? (nb ? 0x7fff : bb) : nb ? ab : a > b ? ab : b > a ? bb : (ab & 0x8000) ? bb : ab;
        uint16_t mn = na ? (nb ? 0x7fff : bb) : nb ? ab : a < b ? ab : b < a ? bb : (ab & 0x8000) ? ab : bb;
        expect16(i, 7, mx);
        expect16(i, 8, mn);
        flag(i, 9, unsigned(a < b) | unsigned(a == b) << 1 | unsigned(!(a == b)) << 2 | unsigned(!(a < b)) << 3 |
                       unsigned(na) << 4);
        expect16(i, 10, fm.round(fin[i], 0));
        expect16(i, 11, fm.round(din[i], 0));
        expect16(i, 12, fm.round(iin[i], 0));
        // To int toward zero, saturating, NaN -> 0.
        auto sat = [&](double v, double lo, double hi) -> long long {
            if (std::isnan(v)) return 0;
            v = std::trunc(v);
            return v <= lo ? (long long)lo : v >= hi ? (long long)hi : (long long)v;
        };
        flag(i, 13, (unsigned short)sat(a, -2147483648.0, 2147483647.0));
        flag(i, 14, (unsigned short)sat(a, -32768, 32767));
        // fma2(p, q, p) with p = (a, b), q = (c, a); only where double is exact.
        if (std::fabs(a) <= 64 && std::fabs(b) <= 64 && std::fabs(c) <= 64) {
            expect16(i, 15, fm.round(a * c + a, 0));
            expect16(i, 16, fm.round(b * a + b, 0));
        }
        double acc = fm.to_double(fm.round(a + b, 0));
        expect16(i, 17, fm.round(acc * c, 0));
        // Shuffles inside the warp of 32 consecutive threads (block of 96).
        int lane = i % 32, base = i - lane;
        int src = base + (lane ^ 3);
        if (src < n) expect16(i, 18, in[3 * src]);
        int down = lane + 1 < 32 && i + 1 < n ? i + 1 : i;
        expect16(i, 19, in[3 * down + 1]);
        double ms = fm.to_double(fm.round(a * b, 0));
        expect16(i, 20, std::isnan(ms) || ms <= 0 ? 0 : ms >= 1 ? fm.round(1.0, 0) : fm.round(a * b, 0));
        bool gt = a > c && b > a, lt = a < c && b < a;
        flag(i, 21, unsigned(gt) | unsigned(lt) << 1);
        expect16(i, 22, fm.round(fin[i], 1));
        flag(i, 23, (unsigned short)(fm.round(fin[i], 2) ^ (fm.round(fin[i], 3) << 1)));
    }
    cudaFree(d_in), cudaFree(d_out), cudaFree(d_f), cudaFree(d_d), cudaFree(d_i);
    return 0;
}

int main() {
    Checker check("types");
    const Format half_fmt(false), bf16_fmt(true);
    check16("half", half_fmt, true, check);
    check16("bf16", bf16_fmt, false, check);

    // Half block sums (fixed tree order) and a gather of 2-byte elements.
    {
        const int blocks = 40, n = blocks * 128;
        std::vector<__half> x(n);
        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i) {
            x[i] = __ushort_as_half(half_fmt.round((int(rnd() % 2001) - 1000) / 256.0, 0));
            idx[i] = int(rnd() % n);
        }
        __half *d_x, *d_s, *d_g;
        int* d_idx;
        CHECK_CUDA(cudaMalloc(&d_x, n * 2));
        CHECK_CUDA(cudaMalloc(&d_s, blocks * 2));
        CHECK_CUDA(cudaMalloc(&d_g, n * 2));
        CHECK_CUDA(cudaMalloc(&d_idx, n * 4));
        CHECK_CUDA(cudaMemcpy(d_x, x.data(), n * 2, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_idx, idx.data(), n * 4, cudaMemcpyHostToDevice));
        half_reduce<<<blocks, 128>>>(d_x, d_idx, d_s, d_g);
        std::vector<__half> s(blocks), g(n);
        CHECK_CUDA(cudaMemcpy(s.data(), d_s, blocks * 2, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(g.data(), d_g, n * 2, cudaMemcpyDeviceToHost));
        for (int b = 0; b < blocks; ++b) {
            uint16_t v[128];
            for (int t = 0; t < 128; ++t) v[t] = x[b * 128 + t].__x;
            for (int stride = 64; stride > 0; stride >>= 1)
                for (int t = 0; t < stride; ++t) v[t] = half_fmt.round(half_fmt.to_double(v[t]) + half_fmt.to_double(v[t + stride]), 0);
            check.expect(s[b].__x == v[0], "half block sum", b, s[b].__x, v[0]);
        }
        for (int i = 0; i < n; ++i) check.expect(g[i].__x == x[idx[i]].__x, "half gather", i, g[i].__x, x[idx[i]].__x);
        cudaFree(d_x), cudaFree(d_s), cudaFree(d_g), cudaFree(d_idx);
    }

    // Small integers.
    {
        const int n = 4000;
        std::vector<signed char> a8(n);
        std::vector<unsigned char> b8(n);
        std::vector<short> a16(n);
        std::vector<unsigned short> b16(n);
        std::vector<char4> c4(n);
        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i) {
            a8[i] = (signed char)rnd(), b8[i] = (unsigned char)rnd();
            a16[i] = (short)rnd(), b16[i] = (unsigned short)rnd();
            uint32_t w = rnd();
            std::memcpy(&c4[i], &w, 4);
            idx[i] = int(rnd() % n);
        }
        if (n > 7) a8[7] = -128, a8[idx[7]] = -1;  // the one overflowing int8 division (promoted: fine)
        signed char *d_a8;
        unsigned char* d_b8;
        short* d_a16;
        unsigned short* d_b16;
        char4* d_c4;
        int* d_idx;
        IntOut* d_out;
        CHECK_CUDA(cudaMalloc(&d_a8, n));
        CHECK_CUDA(cudaMalloc(&d_b8, n));
        CHECK_CUDA(cudaMalloc(&d_a16, n * 2));
        CHECK_CUDA(cudaMalloc(&d_b16, n * 2));
        CHECK_CUDA(cudaMalloc(&d_c4, n * 4));
        CHECK_CUDA(cudaMalloc(&d_idx, n * 4));
        CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(IntOut)));
        CHECK_CUDA(cudaMemcpy(d_a8, a8.data(), n, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b8, b8.data(), n, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_a16, a16.data(), n * 2, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b16, b16.data(), n * 2, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_c4, c4.data(), n * 4, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_idx, idx.data(), n * 4, cudaMemcpyHostToDevice));
        const signed char k8 = -77;
        const short k16 = 12345;
        small_ints<<<(n + 127) / 128, 128>>>(d_a8, d_b8, d_a16, d_b16, d_c4, d_idx, d_out, k8, k16, n);
        CHECK_CUDA(cudaGetLastError());
        std::vector<IntOut> out(n);
        CHECK_CUDA(cudaMemcpy(out.data(), d_out, n * sizeof(IntOut), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            int j = idx[i];
            signed char x = a8[i], y = a8[j];
            unsigned char u = b8[i], v = b8[j];
            short p = a16[i], q = a16[j];
            unsigned short r = b16[i], s = b16[j];
            IntOut w{};
            w.s8[0] = (signed char)(x * y + k8);
            w.s8[1] = (signed char)(y != 0 ? x / y : x);
            w.s8[2] = (signed char)(x >> 2) ^ (signed char)(y << 3);
            w.s8[3] = x < y ? x : y;
            w.u8[0] = (unsigned char)(u * v + 7);
            w.u8[1] = (unsigned char)(v ? u % v : u);
            w.u8[2] = (unsigned char)(u - v);
            w.u8[3] = (unsigned char)((u + v) >> 1);
            w.s16[0] = (short)(p * q + k16);
            w.s16[1] = (short)(q != 0 ? p / q : p);
            w.s16[2] = (short)((p << 3) ^ (q >> 1));
            w.s16[3] = p > q ? p : q;
            w.u16[0] = (unsigned short)(r * s + 1);
            w.u16[1] = (unsigned short)(s ? r % s : r);
            w.u16[2] = (unsigned short)(r - s);
            w.u16[3] = (unsigned short)(r >> 5);
            const signed char* wb = &c4[i].x;
            const signed char* zb = &c4[j].x;
            uint32_t wa, za;
            std::memcpy(&wa, wb, 4), std::memcpy(&za, zb, 4);
            int dp4 = i;
            unsigned udp4 = 5;
            for (int k = 0; k < 4; ++k) dp4 += wb[k] * zb[k], udp4 += unsigned((uint8_t)wb[k]) * (uint8_t)zb[k];
            w.dp[0] = dp4;
            w.dp[1] = int(udp4);
            w.dp[2] = -3 + int(int16_t(wa)) * zb[0] + int(int16_t(wa >> 16)) * zb[1];
            w.dp[3] = 9 + int(int16_t(wa)) * zb[2] + int(int16_t(wa >> 16)) * zb[3];
            w.dp[4] = c4[i].x * c4[j].w + c4[i].y * c4[j].z;
            unsigned sad = 0;
            uint32_t vadd4 = 0, vsub2 = 0, vabs = 0, vmax = 0, veq = 0, vavg = 0;
            for (int k = 0; k < 4; ++k) {
                uint8_t x8 = uint8_t(wa >> (8 * k)), z8 = uint8_t(za >> (8 * k));
                sad += x8 > z8 ? x8 - z8 : z8 - x8;
                vadd4 |= uint32_t(uint8_t(x8 + z8)) << (8 * k);
                vabs |= uint32_t(uint8_t(x8 > z8 ? x8 - z8 : z8 - x8)) << (8 * k);
                vmax |= uint32_t(uint8_t(std::max(int8_t(x8), int8_t(z8)))) << (8 * k);
                uint8_t e8 = uint8_t((wa ^ 0x00ff0000u) >> (8 * k));
                veq |= uint32_t(x8 == e8 ? 0xff : 0) << (8 * k);
            }
            for (int k = 0; k < 2; ++k) {
                uint16_t x16 = uint16_t(wa >> (16 * k)), z16 = uint16_t(za >> (16 * k));
                vsub2 |= uint32_t(uint16_t(x16 - z16)) << (16 * k);
                vavg |= uint32_t(uint16_t((unsigned(x16) + z16 + 1) >> 1)) << (16 * k);
            }
            w.dp[5] = int(sad);
            w.simd[0] = vadd4, w.simd[1] = vsub2, w.simd[2] = vabs, w.simd[3] = vmax, w.simd[4] = veq, w.simd[5] = vavg;
            const IntOut& g = out[i];
            for (int k = 0; k < 4; ++k) {
                check.expect(g.s8[k] == w.s8[k], "int8", i * 4 + k, g.s8[k], w.s8[k]);
                check.expect(g.u8[k] == w.u8[k], "uint8", i * 4 + k, g.u8[k], w.u8[k]);
                check.expect(g.s16[k] == w.s16[k], "int16", i * 4 + k, g.s16[k], w.s16[k]);
                check.expect(g.u16[k] == w.u16[k], "uint16", i * 4 + k, g.u16[k], w.u16[k]);
            }
            for (int k = 0; k < 6; ++k) {
                check.expect(g.dp[k] == w.dp[k], "dp4a/dp2a/sad", i * 6 + k, g.dp[k], w.dp[k]);
                check.expect(g.simd[k] == w.simd[k], "simd in word", i * 6 + k, g.simd[k], w.simd[k]);
            }
        }
        cudaFree(d_a8), cudaFree(d_b8), cudaFree(d_a16), cudaFree(d_b16), cudaFree(d_c4), cudaFree(d_idx), cudaFree(d_out);
    }

    // 32/64-bit integers and fp32/fp64.
    {
        const int n = 3000;
        std::vector<long long> a(n);
        std::vector<double> d(n);
        for (int i = 0; i < n; ++i) {
            a[i] = (long long)(((uint64_t)rnd() << 32) | rnd()) >> (rnd() % 40);
            d[i] = std::ldexp(1.0 + (rnd() % 1000003) / 1000003.0, int(rnd() % 60) - 30) * ((rnd() & 1) ? -1 : 1);
        }
        long long* d_a;
        double* d_d;
        WideOut* d_out;
        CHECK_CUDA(cudaMalloc(&d_a, n * 8));
        CHECK_CUDA(cudaMalloc(&d_d, n * 8));
        CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(WideOut)));
        CHECK_CUDA(cudaMemcpy(d_a, a.data(), n * 8, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_d, d.data(), n * 8, cudaMemcpyHostToDevice));
        wide<<<(n + 63) / 64, 64>>>(d_a, d_d, d_out, n);
        std::vector<WideOut> out(n);
        CHECK_CUDA(cudaMemcpy(out.data(), d_out, n * sizeof(WideOut), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            long long x = a[i], y = a[(i * 13 + 5) % n] | 1;
            double p = d[i], q = d[(i * 7 + 1) % n];
            const WideOut& g = out[i];
            check.expect(g.i32[0] == int(unsigned(int(x)) * unsigned(int(y))), "int32 mul", i, g.i32[0], 0);
            check.expect(g.i32[1] == int(x / int(y)), "int32 div", i, g.i32[1], int(x / int(y)));
            check.expect(g.i32[2] == int(std::nearbyint(float(p))), "float2int_rn", i, g.i32[2], std::nearbyint(float(p)));
            check.expect(g.u32[0] == unsigned(x) >> (i & 31), "uint32 shift", i, g.u32[0], unsigned(x) >> (i & 31));
            check.expect(g.u32[1] == unsigned(x) % (unsigned(y) | 1u), "uint32 rem", i, g.u32[1], 0);
            check.expect(g.i64[0] == (long long)((unsigned long long)x * (unsigned long long)y), "int64 mul", i, g.i64[0], 0);
            check.expect(g.i64[1] == x / y, "int64 div", i, g.i64[1], x / y);
            check.expect(g.i64[2] == x % y, "int64 rem", i, g.i64[2], x % y);
            check.expect(g.i64[3] == (long long)(((__int128)x * y) >> 64), "mul64hi", i, g.i64[3], 0);
            check.expect(g.u64[0] == (unsigned long long)x * (unsigned long long)y, "uint64 mul", i, g.u64[0], 0);
            check.expect(g.u64[1] == (unsigned long long)x / ((unsigned long long)y | 1ull), "uint64 div", i, g.u64[1], 0);
            check.expect(g.u64[2] == (unsigned long long)x >> (i & 63), "uint64 shift", i, g.u64[2], 0);
            check.expect(g.f32[0] == float(x), "int64 -> float", i, g.f32[0], float(x));
            // Device code contracts a*b + 1 into an fma (fmad on by default).
            float ff = std::fma(float(p), float(q), 1.0f);
            check.expect(g.f32[1] == ff, "float fma", i, g.f32[1], ff);
            check.expect(g.f32[2] == float(p / q), "double -> float", i, g.f32[2], float(p / q));
            check.expect(g.f64[0] == double(x), "int64 -> double", i, g.f64[0], double(x));
            check.expect(g.f64[1] == std::fma(p, q, -p), "double fma", i, g.f64[1], std::fma(p, q, -p));
            check.expect(g.f64[2] == p / q, "double div", i, g.f64[2], p / q);
        }
        cudaFree(d_a), cudaFree(d_d), cudaFree(d_out);
    }

    // fma ties broken by a tiny addend.
    {
        float* d_out;
        CHECK_CUDA(cudaMalloc(&d_out, 4 * 4));
        fma_ties<<<1, 1>>>(d_out);
        float got[4];
        CHECK_CUDA(cudaMemcpy(got, d_out, sizeof got, cudaMemcpyDeviceToHost));
        // bf16: 1.51171875 is halfway between 1.5078125 and 1.515625; half:
        // 1.50146484375 between 1.5009765625 and 1.501953125.
        const float want[4] = {1.5078125f, 1.515625f, 1.5009765625f, 1.501953125f};
        for (int k = 0; k < 4; ++k) check.expect(got[k] == want[k], "fma tie decided by c", k, got[k], want[k]);
        cudaFree(d_out);
    }

    // Small types as kernel arguments.
    {
        float* d_out;
        CHECK_CUDA(cudaMalloc(&d_out, 7 * 4));
        args<<<1, 1>>>(__float2half(-2.5f), __float2bfloat16(3.25f), (signed char)-100, (short)-30000, (unsigned char)250,
                       1e10, -(1ll << 40), d_out);
        float got[7];
        CHECK_CUDA(cudaMemcpy(got, d_out, sizeof got, cudaMemcpyDeviceToHost));
        const float want[7] = {-2.5f, 3.25f, -100, -30000, 250, 1e10f, -1099511627776.0f};
        for (int k = 0; k < 7; ++k) check.expect(got[k] == want[k], "kernel argument", k, got[k], want[k]);
        cudaFree(d_out);
    }
    return check.finish();
}
