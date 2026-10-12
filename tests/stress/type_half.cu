// 16-bit floating point: __half and __nv_bfloat16 conversions (every bit
// pattern), arithmetic intrinsics and operators, __half2/__nv_bfloat162
// packed ops (__hadd2, __hmul2, __hfma2...), half-precision accumulation.
// Host references are computed in double, exact for these operand ranges,
// and rounded once with __double2half / __double2bfloat16.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "stress.cuh"

static inline unsigned short hbits(__half h) { return __half_as_ushort(h); }
static inline unsigned short bbits(__nv_bfloat16 h) { return __bfloat16_as_ushort(h); }
static inline bool h_nan(unsigned short b) { return (b & 0x7c00) == 0x7c00 && (b & 0x3ff); }
static inline bool b_nan(unsigned short b) { return (b & 0x7f80) == 0x7f80 && (b & 0x7f); }

// Bits equal, or both NaN (payloads are not specified).
static bool same16(const std::vector<unsigned short>& got, const std::vector<unsigned short>& want, bool bf, const char* what,
                   int ulps = 0) {
    for (size_t i = 0; i < want.size(); ++i) {
        unsigned short g = got[i], w = want[i];
        bool gn = bf ? b_nan(g) : h_nan(g), wn = bf ? b_nan(w) : h_nan(w);
        if (gn || wn) {
            if (gn != wn) goto bad;
            continue;
        }
        if (g == w) continue;
        if (ulps && ((g ^ w) & 0x8000) == 0 && std::abs(int(g) - int(w)) <= ulps) continue;
    bad:
        std::printf("  mismatch %s[%zu]: got 0x%04x want 0x%04x\n", what, i, g, w);
        return false;
    }
    return true;
}

// Every half bit pattern to float and back, to int, and every float in a sweep to half.
__global__ void k_half_convert(const float* f, unsigned short* f2h, float* h2f, unsigned short* h2h, int* h2i, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) f2h[i] = __half_as_ushort(__float2half_rn(f[i]));
    if (i < 65536) {
        __half h = __ushort_as_half((unsigned short)i);
        float v = __half2float(h);
        h2f[i] = v;
        h2h[i] = __half_as_ushort(__float2half(v));
        h2i[i] = __hisinf(h) || __hisnan(h) ? 0 : __half2int_rn(h);
    }
}

// Scalar half arithmetic over operand pairs: intrinsics and operators.
__global__ void k_half_arith(const unsigned short* a, const unsigned short* b, const unsigned short* c, unsigned short* out,
                             int* cmp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __half x = __ushort_as_half(a[i]), y = __ushort_as_half(b[i]), z = __ushort_as_half(c[i]);
    out[8 * i + 0] = __half_as_ushort(__hadd(x, y));
    out[8 * i + 1] = __half_as_ushort(__hsub(x, y));
    out[8 * i + 2] = __half_as_ushort(__hmul(x, y));
    out[8 * i + 3] = __half_as_ushort(__hfma(x, y, z));
    out[8 * i + 4] = __half_as_ushort(x * y + z);  // operators (may contract to fma: compared as fma or mul+add)
    out[8 * i + 5] = __half_as_ushort(-x);
    out[8 * i + 6] = __half_as_ushort(__habs(y));
    out[8 * i + 7] = __half_as_ushort(__hdiv(x, y));
    cmp[i] = __hlt(x, y) + 2 * __heq(x, y) + 4 * (x >= y) + 8 * __hisnan(x);
}

// Packed half2.
__global__ void k_half2(const __half2* a, const __half2* b, const __half2* c, __half2* out, float2* f2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __half2 x = a[i], y = b[i], z = c[i];
    out[4 * i + 0] = __hadd2(x, y);
    out[4 * i + 1] = __hmul2(x, y);
    out[4 * i + 2] = __hfma2(x, y, z);
    out[4 * i + 3] = __halves2half2(__high2half(x), __low2half(__hsub2(y, z)));
    f2[i] = __half22float2(__floats2half2_rn(__low2float(x) * 3.0f, __high2float(y) + 1.0f));
}

// bfloat16 scalar and packed.
__global__ void k_bf16(const float* fa, const float* fb, const float* fc, unsigned short* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __nv_bfloat16 x = __float2bfloat16(fa[i]), y = __float2bfloat16(fb[i]), z = __float2bfloat16(fc[i]);
    out[8 * i + 0] = __bfloat16_as_ushort(x);
    out[8 * i + 1] = __bfloat16_as_ushort(__hadd(x, y));
    out[8 * i + 2] = __bfloat16_as_ushort(__hmul(x, y));
    out[8 * i + 3] = __bfloat16_as_ushort(__hfma(x, y, z));
    out[8 * i + 4] = __bfloat16_as_ushort(__float2bfloat16(__bfloat162float(x) * 0.5f));
    __nv_bfloat162 p = __floats2bfloat162_rn(fa[i], fb[i]), q = __halves2bfloat162(z, x);
    __nv_bfloat162 s = __hadd2(p, q), f = __hfma2(p, q, s);
    out[8 * i + 5] = __bfloat16_as_ushort(__low2bfloat16(s));
    out[8 * i + 6] = __bfloat16_as_ushort(__high2bfloat16(f));
    out[8 * i + 7] = __bfloat16_as_ushort(__hsub(x, z));
}

// A dot product of 64 elements accumulated in half with __hfma (rounding at every step).
__global__ void k_half_dot(const __half* a, const __half* b, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __half acc = __float2half(0.0f);
    for (int k = 0; k < 64; ++k) acc = __hfma(a[(i + k) % n], b[(i * 5 + k) % n], acc);
    out[i] = acc;
}

static double hd(unsigned short b) { return double(__half2float(__ushort_as_half(b))); }
static unsigned short h_of(double v) { return hbits(__double2half(v)); }
static double bd(__nv_bfloat16 b) { return double(__bfloat162float(b)); }
static unsigned short b_of(double v) { return bbits(__double2bfloat16(v)); }

int main() {
    using namespace stress;

    // Conversions.
    {
        std::vector<float> f = random_floats(1 << 18, 1, -70000.f, 70000.f);
        auto sm = random_floats(1 << 16, 2, -1e-4f, 1e-4f);  // half denormal range
        f.insert(f.end(), sm.begin(), sm.end());
        float sp[] = {0.f, -0.f, INFINITY, -INFINITY, NAN, 65504.f, 65519.99f, 65520.f, -65520.f, 5.96e-8f, 2.98e-8f, 2.99e-8f,
                      1e-10f, 1.0009765625f, 1.00048828125f, 1.00146484375f, 2049.f, 2051.f, -2049.f};
        f.insert(f.end(), std::begin(sp), std::end(sp));
        int n = int(f.size());
        std::vector<unsigned short> wf2h(n), wh2h(65536);
        std::vector<float> wh2f(65536);
        std::vector<int> wh2i(65536);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) wf2h[i] = h_of(double(f[i]));
#pragma omp parallel for
            for (int i = 0; i < 65536; ++i) {
                double v = hd((unsigned short)i);
                wh2f[i] = float(v);
                wh2h[i] = (unsigned short)i;
                wh2i[i] = std::isinf(v) || std::isnan(v) ? 0 : int(std::nearbyint(v));
            }
        });
        Buffer<float> df(f), dh2f(65536);
        Buffer<unsigned short> df2h(n), dh2h(65536);
        Buffer<int> dh2i(65536);
        double k = time_device([&] { k_half_convert<<<(n + 255) / 256, 256>>>(df, df2h, dh2f, dh2h, dh2i, n); });
        auto g2f = dh2f.download();
        bool ok = same16(df2h.download(), wf2h, false, "f2h") && same16(dh2h.download(), wh2h, false, "roundtrip") &&
                  equal(dh2i.download().data(), wh2i.data(), 65536, "h2i");
        for (int i = 0; i < 65536 && ok; ++i)
            if (!(std::isnan(wh2f[i]) ? std::isnan(g2f[i]) : std::memcmp(&g2f[i], &wh2f[i], 4) == 0))
                ok = false, std::printf("  mismatch h2f[%d]\n", i);
        report("half_conversions_all_patterns", ok, k, ref);
    }

    // Scalar half arithmetic: random patterns in [-8, 8] plus special values.
    {
        const int n = 1 << 17;
        auto fa = random_floats(n, 3, -8, 8), fb = random_floats(n, 4, -8, 8), fc = random_floats(n, 5, -8, 8);
        auto r = random_u32(n, 6);
        std::vector<unsigned short> a(n), b(n), c(n);
        const unsigned short special[] = {0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0x0001, 0x8001, 0x03ff, 0x0400, 0x7bff, 0xfbff, 0x3c00};
        for (int i = 0; i < n; ++i) {
            a[i] = h_of(fa[i]), b[i] = h_of(fb[i]), c[i] = h_of(fc[i]);
            if (r[i] % 8 == 0) a[i] = special[(r[i] >> 3) % 12];
            if (r[i] % 9 == 0) b[i] = special[(r[i] >> 5) % 12];
            if (r[i] % 10 == 0) c[i] = special[(r[i] >> 7) % 12];
        }
        std::vector<unsigned short> want(8 * size_t(n));
        std::vector<int> wcmp(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                double x = hd(a[i]), y = hd(b[i]), z = hd(c[i]);
                want[8 * i + 0] = h_of(x + y);
                want[8 * i + 1] = h_of(x - y);
                want[8 * i + 2] = h_of(x * y);
                want[8 * i + 3] = h_of(x * y + z);
                want[8 * i + 4] = h_of(x * y + z);
                want[8 * i + 5] = a[i] ^ 0x8000;
                want[8 * i + 6] = b[i] & 0x7fff;
                want[8 * i + 7] = h_of(x / y);
                wcmp[i] = (x < y) + 2 * (x == y) + 4 * (x >= y) + 8 * std::isnan(x);
            }
        });
        Buffer<unsigned short> da(a), db(b), dc(c), dout(8 * size_t(n));
        Buffer<int> dcmp(n);
        double k = time_device([&] { k_half_arith<<<(n + 255) / 256, 256>>>(da, db, dc, dout, dcmp, n); });
        auto got = dout.download();
        // Slot 4 (operators x*y+z) may be mul then add (two roundings): accept either.
        for (int i = 0; i < n; ++i) {
            unsigned short two = h_of(hd(h_of(hd(a[i]) * hd(b[i]))) + hd(c[i]));
            if (got[8 * i + 4] == two) want[8 * i + 4] = two;
        }
        // -NaN / |NaN| keep their payload bits on both sides: compared as NaN anyway.
        bool ok = same16(got, want, false, "arith") && equal(dcmp.download().data(), wcmp.data(), n, "cmp");
        report("half_scalar_arith", ok, k, ref);
    }

    // half2.
    {
        const int n = 1 << 17;
        auto f = random_floats(6 * size_t(n), 7, -4, 4);
        std::vector<__half2> a(n), b(n), c(n);
        for (int i = 0; i < n; ++i) {
            a[i] = __halves2half2(__double2half(f[6 * i]), __double2half(f[6 * i + 1]));
            b[i] = __halves2half2(__double2half(f[6 * i + 2]), __double2half(f[6 * i + 3]));
            c[i] = __halves2half2(__double2half(f[6 * i + 4]), __double2half(f[6 * i + 5]));
        }
        std::vector<unsigned short> want(8 * size_t(n));
        std::vector<float> wf(2 * size_t(n));
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                double xl = double(__low2float(a[i])), xh = double(__high2float(a[i]));
                double yl = double(__low2float(b[i])), yh = double(__high2float(b[i]));
                double zl = double(__low2float(c[i])), zh = double(__high2float(c[i]));
                unsigned short* w = &want[8 * size_t(i)];
                w[0] = h_of(xl + yl), w[1] = h_of(xh + yh);
                w[2] = h_of(xl * yl), w[3] = h_of(xh * yh);
                w[4] = h_of(xl * yl + zl), w[5] = h_of(xh * yh + zh);
                w[6] = h_of(xh), w[7] = h_of(yl - zl);  // low = high(x), high = low(y - z)
                wf[2 * i] = float(hd(h_of(double(float(xl) * 3.0f))));
                wf[2 * i + 1] = float(hd(h_of(double(float(yh) + 1.0f))));
            }
        });
        Buffer<__half2> da(a), db(b), dc(c), dout(4 * size_t(n));
        Buffer<float2> df(n);
        double k = time_device([&] { k_half2<<<(n + 255) / 256, 256>>>(da, db, dc, dout, df, n); });
        auto g = dout.download();
        std::vector<unsigned short> gb(8 * size_t(n));
        std::memcpy(gb.data(), g.data(), gb.size() * 2);
        auto gf = df.download();
        bool ok = same16(gb, want, false, "half2") && equal(reinterpret_cast<float*>(gf.data()), wf.data(), 2 * size_t(n), "f2");
        report("half2_packed_ops", ok, k, ref);
    }

    // bfloat16.
    {
        const int n = 1 << 17;
        auto fa = random_floats(n, 8, -100, 100), fb = random_floats(n, 9, -100, 100), fc = random_floats(n, 10, -100, 100);
        fa[0] = INFINITY, fb[1] = NAN, fa[2] = 1e-39f, fb[3] = -3e38f, fa[3] = 3e38f, fc[4] = -0.0f;
        std::vector<unsigned short> want(8 * size_t(n));
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                __nv_bfloat16 bx = __float2bfloat16(fa[i]), by = __float2bfloat16(fb[i]), bz = __float2bfloat16(fc[i]);
                double x = bd(bx), y = bd(by), z = bd(bz);
                unsigned short* w = &want[8 * size_t(i)];
                w[0] = bbits(bx);
                w[1] = b_of(x + y);
                w[2] = b_of(x * y);
                w[3] = b_of(x * y + z);
                w[4] = b_of(double(float(x) * 0.5f));
                double pl = bd(__float2bfloat16(fa[i])), ph = bd(__float2bfloat16(fb[i]));
                double sl = bd(__double2bfloat16(pl + z)), sh = bd(__double2bfloat16(ph + x));
                w[5] = b_of(sl);
                w[6] = b_of(ph * x + sh);
                w[7] = b_of(x - z);
            }
        });
        Buffer<float> da(fa), db(fb), dc(fc);
        Buffer<unsigned short> dout(8 * size_t(n));
        double k = time_device([&] { k_bf16<<<(n + 255) / 256, 256>>>(da, db, dc, dout, n); });
        report("bfloat16_scalar_packed", same16(dout.download(), want, true, "bf16"), k, ref);
    }

    // Half accumulation, 64 rounded fma steps per thread.
    {
        const int n = 1 << 16;
        auto fa = random_floats(n, 11, -1, 1), fb = random_floats(n, 12, -1, 1);
        std::vector<__half> a(n), b(n);
        for (int i = 0; i < n; ++i) a[i] = __double2half(fa[i]), b[i] = __double2half(fb[i]);
        std::vector<unsigned short> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                unsigned short acc = 0;
                for (int k = 0; k < 64; ++k)
                    acc = h_of(double(__half2float(a[(i + k) % n])) * double(__half2float(b[(i * 5 + k) % n])) + hd(acc));
                want[i] = acc;
            }
        });
        Buffer<__half> da(a), db(b), dout(n);
        double k = time_device([&] { k_half_dot<<<(n + 255) / 256, 256>>>(da, db, dout, n); });
        auto g = dout.download();
        std::vector<unsigned short> gb(n);
        std::memcpy(gb.data(), g.data(), 2 * size_t(n));
        report("half_fma_accumulate_64", same16(gb, want, false, "dot"), k, ref);
    }
    return finish();
}
