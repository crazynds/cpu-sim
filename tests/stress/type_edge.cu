// Numeric edge cases. Some threads do things whose result C++ leaves
// undefined or unspecified (integer division by zero, INT_MIN / -1, shifts by
// 32 or more, out-of-range float to int): the GPU does not trap, so those
// cases only check that the kernel completes and that every OTHER thread is
// right. Well-defined edge cases (NaN propagation, signed zeros, denormals
// without flush to zero, volatile locals) are checked exactly.
#include <climits>

#include "stress.cuh"

// Integer division with a zero divisor in some lanes (data dependent).
__global__ void k_divzero(const int* a, const int* b, int* q, int* r, unsigned* uq, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int x = a[i], y = b[i];
    q[i] = x / y;
    r[i] = x % y;
    uq[i] = unsigned(x) / unsigned(y);
}

// INT_MIN / -1 and INT_MIN % -1 (also the 64-bit forms) in some lanes.
__global__ void k_intmin(const int* a, const int* b, const long long* la, const long long* lb, int* q, long long* lq, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    q[i] = a[i] / b[i] + a[i] % b[i];
    lq[i] = la[i] / lb[i] + la[i] % lb[i];
}

// Shifts by a data-dependent amount in [0, 40): lanes with s >= 32 are UB.
__global__ void k_shift(const unsigned* x, const int* s, unsigned* l, unsigned* r, int* ar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    l[i] = x[i] << s[i];
    r[i] = x[i] >> s[i];
    ar[i] = int(x[i]) >> s[i];
}

// Out-of-range and NaN float to int conversions (UB in C++): completion only.
__global__ void k_f2i_range(const float* f, int* i32, unsigned* u32, long long* i64, short* i16, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    i32[i] = int(f[i]);
    u32[i] = unsigned(f[i]);
    i64[i] = (long long)(f[i] * 1e10f);
    i16[i] = short(f[i]);
}

// NaN propagation, comparisons and min/max (exact).
__host__ __device__ inline unsigned nan_rules(float a, float b) {
    unsigned r = 0;
    r |= isnan(a + b) << 0;
    r |= isnan(a * 0.0f) << 1;
    r |= (a < b) << 2;
    r |= (a == a) << 3;
    r |= (a != b) << 4;
    r |= isnan(fminf(a, b)) << 5;
    r |= isnan(fmaxf(a, b)) << 6;
    r |= isnan(sqrtf(a)) << 7;
    r |= isnan(a - a) << 8;
    r |= (!(a >= b)) << 9;
    r |= isinf(a / b) << 10;
    float m = fminf(a, b);
    r |= (m == b) << 11;
    r |= (fmaxf(a, b) == a) << 12;
    r |= isnan(b / a) << 13;
    return r;
}
__global__ void k_nan(const float* a, const float* b, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = nan_rules(a[i], b[i]);
}

// Signed zeros.
__host__ __device__ inline unsigned zero_rules(float z, float x) {
    unsigned r = 0;
    float nz = -z;
    r |= signbit(z * -1.0f) << 0;
    r |= signbit(nz) << 1;
    r |= (nz == z) << 2;
    r |= signbit(z + nz) << 3;   // +0 in round to nearest
    r |= signbit(nz - z) << 4;   // -0 - (+0) = -0 for z=+0
    r |= (1.0f / nz < 0) << 5;   // -inf
    r |= signbit(copysignf(x, nz)) << 6;
    r |= signbit(x * nz) << 7;
    r |= signbit(sqrtf(nz)) << 8;  // sqrt(-0) = -0
    r |= signbit(nz * nz) << 9;
    r |= signbit(fabsf(nz)) << 10;
    r |= signbit(floorf(-0.25f * fabsf(x) - 0.0f) + 0.0f) << 11;
    r |= signbit(ceilf(-0.25f)) << 12;  // ceil(-0.25) = -0
    r |= signbit(truncf(-0.75f * (x != 0))) << 13;
    return r;
}
__global__ void k_zero(const float* z, const float* x, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = zero_rules(z[i], x[i]);
}

// Denormals in and out: single operations, IEEE (no flush to zero).
__global__ void k_denorm(const float* a, const float* b, float* out, const double* da, double* dout, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i], y = b[i];
    out[6 * i + 0] = x * y;
    out[6 * i + 1] = x + y;
    out[6 * i + 2] = x - y;
    out[6 * i + 3] = x / (y * 4096.0f + 1.0f);
    out[6 * i + 4] = sqrtf(fabsf(x));
    out[6 * i + 5] = x * 0x1p-20f;
    double dx = da[i];
    dout[2 * i] = dx * 0x1p-30;
    dout[2 * i + 1] = dx + da[(i + 1) % n];
}

// volatile locals: every access is a real load/store, results unchanged.
__global__ void k_volatile(const int* in, int* out, float* fout, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    volatile int acc = 0;
    volatile float facc = 0.0f;
    volatile int idx[4] = {0, 1, 2, 3};
    for (int k = 0; k < 32; ++k) {
        acc = acc + in[(i + k) % n] * idx[k & 3];
        facc = facc + float(in[(i * 3 + k) % n] & 0xff) * 0.5f;
        idx[k & 3] = idx[k & 3] + 1;
    }
    out[i] = acc;
    fout[i] = facc;
}

int main() {
    using namespace stress;
    const int n = 1 << 18, B = 256, G = (n + B - 1) / B;
    auto r = random_u32(4 * size_t(n), 77);

    {
        std::vector<int> a(n), b(n);
        for (int i = 0; i < n; ++i) {
            a[i] = int(r[i]);
            b[i] = (r[n + i] % 5 == 0) ? 0 : int(r[n + i]) >> (r[2 * n + i] % 31);
            if (b[i] == -1 && a[i] == INT_MIN) a[i] = 1;
        }
        std::vector<int> wq(n), wr(n);
        std::vector<unsigned> wu(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i)
                if (b[i]) wq[i] = a[i] / b[i], wr[i] = a[i] % b[i], wu[i] = unsigned(a[i]) / unsigned(b[i]);
        });
        Buffer<int> da(a), db(b), dq(n), dr(n);
        Buffer<unsigned> du(n);
        double k = time_device([&] { k_divzero<<<G, B>>>(da, db, dq, dr, du, n); });
        auto q = dq.download(), rr = dr.download();
        auto u = du.download();
        bool ok = true;
        for (int i = 0; i < n && ok; ++i)
            if (b[i] && (q[i] != wq[i] || rr[i] != wr[i] || u[i] != wu[i])) ok = false, std::printf("  mismatch at %d\n", i);
        report("int_div_by_zero_some_lanes", ok, k, ref, "zero-divisor lanes: completion only");
    }
    {
        std::vector<int> a(n), b(n), wq(n);
        std::vector<long long> la(n), lb(n), wl(n);
        for (int i = 0; i < n; ++i) {
            bool edge = i % 3 == 0;
            a[i] = edge ? INT_MIN : int(r[i]);
            b[i] = edge ? -1 : (int(r[n + i] % 2001) - 1000) | 1;
            la[i] = edge ? LLONG_MIN : (long long)r[i] << 30;
            lb[i] = edge ? -1 : (long long)(int(r[2 * n + i]) | 1);
        }
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i)
                if (i % 3) wq[i] = a[i] / b[i] + a[i] % b[i], wl[i] = la[i] / lb[i] + la[i] % lb[i];
        });
        Buffer<int> da(a), db(b), dq(n);
        Buffer<long long> dla(la), dlb(lb), dlq(n);
        double k = time_device([&] { k_intmin<<<G, B>>>(da, db, dla, dlb, dq, dlq, n); });
        auto q = dq.download();
        auto lq = dlq.download();
        bool ok = true;
        for (int i = 0; i < n && ok; ++i)
            if (i % 3 && (q[i] != wq[i] || lq[i] != wl[i])) ok = false, std::printf("  mismatch at %d\n", i);
        report("int_min_div_minus_one", ok, k, ref, "INT_MIN/-1 lanes: completion only");
    }
    {
        std::vector<unsigned> x(r.begin(), r.begin() + n), wl(n), wr(n);
        std::vector<int> s(n), war(n);
        for (int i = 0; i < n; ++i) s[i] = int(r[n + i] % 40);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i)
                if (s[i] < 32) wl[i] = x[i] << s[i], wr[i] = x[i] >> s[i], war[i] = int(x[i]) >> s[i];
        });
        Buffer<unsigned> dx(x), dl(n), dr(n);
        Buffer<int> ds(s), dar(n);
        double k = time_device([&] { k_shift<<<G, B>>>(dx, ds, dl, dr, dar, n); });
        auto l = dl.download(), rr = dr.download();
        auto ar = dar.download();
        bool ok = true;
        for (int i = 0; i < n && ok; ++i)
            if (s[i] < 32 && (l[i] != wl[i] || rr[i] != wr[i] || ar[i] != war[i])) ok = false, std::printf("  mismatch at %d\n", i);
        report("shift_by_32_or_more", ok, k, ref, "shift>=32 lanes: completion only");
    }
    {
        auto f = random_floats(n, 5, -1e3f, 1e3f);
        for (int i = 0; i < n; i += 4) {
            const float bad[] = {3e9f, -3e9f, 1e20f, -1e20f, INFINITY, -INFINITY, NAN, 2147483648.f, -2147483904.f, 4294967296.f, 70000.f, -1.f};
            f[i] = bad[(i / 4) % 12];
        }
        Buffer<float> df(f);
        Buffer<int> d32(n);
        Buffer<unsigned> du(n);
        Buffer<long long> d64(n);
        Buffer<short> d16(n);
        double k = time_device([&] { k_f2i_range<<<G, B>>>(df, d32, du, d64, d16, n); });
        auto g32 = d32.download();
        auto g64 = d64.download();
        bool ok = true;
        for (int i = 0; i < n && ok; ++i)
            if (i % 4 && (g32[i] != int(f[i]) || g64[i] != (long long)(f[i] * 1e10f))) ok = false, std::printf("  mismatch at %d\n", i);
        report("float_to_int_out_of_range", ok, k, -1, "out-of-range lanes: completion only");
    }
    {
        auto a = random_floats(n, 6, -10, 10), b = random_floats(n, 7, -10, 10);
        const float sp[] = {NAN, -NAN, INFINITY, -INFINITY, 0.0f, -0.0f, 1.0f};
        for (int i = 0; i < n; ++i) {
            if (r[i] % 3 == 0) a[i] = sp[(r[i] >> 2) % 7];
            if (r[i] % 5 == 0) b[i] = sp[(r[i] >> 5) % 7];
        }
        std::vector<unsigned> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = nan_rules(a[i], b[i]);
        });
        Buffer<float> da(a), db(b);
        Buffer<unsigned> dout(n);
        double k = time_device([&] { k_nan<<<G, B>>>(da, db, dout, n); });
        report("nan_inf_propagation", equal(dout.download().data(), want.data(), n), k, ref);
    }
    {
        std::vector<float> z(n), x = random_floats(n, 8, -5, 5);
        for (int i = 0; i < n; ++i) z[i] = (i & 1) ? 0.0f : -0.0f;
        for (int i = 0; i < n; i += 7) x[i] = 0.0f;
        std::vector<unsigned> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = zero_rules(z[i], x[i]);
        });
        Buffer<float> dz(z), dx(x);
        Buffer<unsigned> dout(n);
        double k = time_device([&] { k_zero<<<G, B>>>(dz, dx, dout, n); });
        report("signed_zero_rules", equal(dout.download().data(), want.data(), n), k, ref);
    }
    {
        std::vector<float> a(n), b(n), want(6 * size_t(n));
        std::vector<double> da(n), dwant(2 * size_t(n));
        auto u = random_floats(2 * size_t(n), 9, 0.5f, 2.0f);
        for (int i = 0; i < n; ++i) {
            a[i] = u[i] * std::ldexp(1.0f, -126 - int(r[i] % 24)) * ((r[i] & 32) ? -1 : 1);  // denormal
            b[i] = u[n + i] * std::ldexp(1.0f, int(r[n + i] % 30) - 140);
            da[i] = double(u[i]) * std::ldexp(1.0, -1022 - int(r[2 * n + i] % 52));
        }
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                float x = a[i], y = b[i];
                want[6 * i + 0] = x * y;
                want[6 * i + 1] = x + y;
                want[6 * i + 2] = x - y;
                want[6 * i + 3] = x / (y * 4096.0f + 1.0f);
                want[6 * i + 4] = std::sqrt(std::fabs(x));
                want[6 * i + 5] = x * 0x1p-20f;
                dwant[2 * i] = da[i] * 0x1p-30;
                dwant[2 * i + 1] = da[i] + da[(i + 1) % n];
            }
        });
        Buffer<float> dA(a), dB(b), dout(6 * size_t(n));
        Buffer<double> dD(da), ddout(2 * size_t(n));
        double k = time_device([&] { k_denorm<<<G, B>>>(dA, dB, dout, dD, ddout, n); });
        bool ok = equal(dout.download().data(), want.data(), 6 * size_t(n), "f") && equal(ddout.download().data(), dwant.data(), 2 * size_t(n), "d");
        auto g = dout.download();
        int nden = 0;
        for (size_t i = 0; i < g.size(); ++i) nden += std::fpclassify(g[i]) == FP_SUBNORMAL;
        if (nden < n) ok = false, std::printf("  only %d denormal outputs\n", nden);
        report("denormals_no_ftz", ok, k, ref);
    }
    {
        std::vector<int> in(n), want(n);
        std::vector<float> fwant(n);
        for (int i = 0; i < n; ++i) in[i] = int(r[i] % 20001) - 10000;
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                int acc = 0, idx[4] = {0, 1, 2, 3};
                float facc = 0;
                for (int k = 0; k < 32; ++k) {
                    acc += in[(i + k) % n] * idx[k & 3];
                    facc += float(in[(i * 3 + k) % n] & 0xff) * 0.5f;
                    idx[k & 3]++;
                }
                want[i] = acc, fwant[i] = facc;
            }
        });
        Buffer<int> din(in), dout(n);
        Buffer<float> dfout(n);
        double k = time_device([&] { k_volatile<<<G, B>>>(din, dout, dfout, n); });
        report("volatile_locals", equal(dout.download().data(), want.data(), n) && equal(dfout.download().data(), fwant.data(), n), k, ref);
    }
    return finish();
}
