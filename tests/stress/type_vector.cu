// CUDA vector types: float2/float4/int2/int4/double2/uchar4/short2/char4/
// uint2/longlong2, make_* constructors, member access, whole-vector loads and
// stores, reinterpreting a scalar array as a vector array.
#include "stress.cuh"

__host__ __device__ inline float4 f4_op(float4 a, float4 b) {
    float4 r;
    r.x = fmaf(a.x, b.y, a.z);
    r.y = a.y - b.x;
    r.z = fmaxf(a.z, b.w) * 0.5f;
    r.w = a.w + b.z + a.x;
    return r;
}
__global__ void k_float4(const float4* a, const float4* b, float4* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f4_op(a[i], b[n - 1 - i]);
}

// float2 complex multiply-accumulate (explicit fma for an exact match).
__host__ __device__ inline float2 cmul(float2 a, float2 b) {
    return make_float2(fmaf(a.x, b.x, -a.y * b.y), fmaf(a.x, b.y, a.y * b.x));
}
__global__ void k_float2(const float2* a, float2* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float2 z = make_float2(0.f, 0.f), c = a[i];
    for (int k = 0; k < 8; ++k) {
        z = cmul(z, z);
        z.x += c.x, z.y += c.y;
    }
    out[i] = z;
}

__host__ __device__ inline int4 i4_op(int4 a, int2 b) {
    return make_int4(a.x + b.x, a.y * b.y, a.z ^ (b.x - b.y), max(a.w, b.y) - min(a.x, b.x));
}
__global__ void k_int4(const int4* a, const int2* b, int4* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i4_op(a[i], b[i]);
}

__host__ __device__ inline double2 d2_op(double2 a, double2 b) {
    double2 r = make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    r.x = fma(r.x, 0.5, a.y);
    return r;
}
__global__ void k_double2(const double2* a, double2* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = d2_op(a[i], a[(i * 7 + 1) % n]);
}

// RGBA pixels: uchar4 to float, blend, back with saturation.
__host__ __device__ inline unsigned char sat8(float v) { return (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v + 0.5f); }
__host__ __device__ inline uchar4 blend(uchar4 p, uchar4 q) {
    float a = q.w / 255.0f;
    return make_uchar4(sat8(p.x * (1 - a) + q.x * a), sat8(p.y * (1 - a) + q.y * a), sat8(p.z * (1 - a) + q.z * a + 3.0f),
                       (unsigned char)(p.w | q.w));
}
__global__ void k_uchar4(const uchar4* p, const uchar4* q, uchar4* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = blend(p[i], q[i]);
}

// short2 and char4: packed small integers, sign extension.
__host__ __device__ inline int small_op(short2 s, char4 c) {
    short2 t = make_short2(short(s.x + c.x), short(s.y - c.w));
    return t.x * 3 + t.y * 5 + c.y * c.z + (s.x >> 3);
}
__global__ void k_small(const short2* s, const char4* c, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = small_op(s[i], c[i]);
}

// uint2 and longlong2.
__host__ __device__ inline longlong2 wide_op(uint2 u, longlong2 l) {
    unsigned long long v = (unsigned long long)u.y << 32 | u.x;
    return make_longlong2(l.x + (long long)v, l.y ^ (long long)(v >> 17));
}
__global__ void k_wide(const uint2* u, const longlong2* l, longlong2* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = wide_op(u[i], l[i]);
}

// A float array read as float4 (vectorised loads) with a scalar tail.
__global__ void k_reinterpret(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;
    if (i < n4) {
        float4 v = reinterpret_cast<const float4*>(x)[i];
        v.x *= 2.f, v.y *= 3.f, v.z *= 4.f, v.w *= 5.f;
        reinterpret_cast<float4*>(y)[i] = v;
    } else if (i - n4 < n % 4) {
        int j = n4 * 4 + (i - n4);
        y[j] = x[j] * float(2 + j % 4);
    }
}

template <class T>
static bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0) {
            std::printf("  mismatch at element %zu\n", i);
            return false;
        }
    return true;
}

template <class T, class Fill>
static std::vector<T> make_vec(int n, Fill f) {
    std::vector<T> v(n);
    for (int i = 0; i < n; ++i) v[i] = f(i);
    return v;
}

int main() {
    using namespace stress;
    const int n = 1 << 17, B = 256, G = (n + B - 1) / B;
    auto f = random_floats(8 * size_t(n), 1);
    auto r = random_u32(4 * size_t(n), 2);

    {
        auto a = make_vec<float4>(n, [&](int i) { return make_float4(f[4 * i], f[4 * i + 1], f[4 * i + 2], f[4 * i + 3]); });
        auto b = make_vec<float4>(n, [&](int i) { return make_float4(f[4 * n + 4 * i], f[4 * n + 4 * i + 1], f[4 * n + 4 * i + 2], f[4 * n + 4 * i + 3]); });
        std::vector<float4> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = f4_op(a[i], b[n - 1 - i]);
        });
        Buffer<float4> da(a), db(b), dout(n);
        double k = time_device([&] { k_float4<<<G, B>>>(da, db, dout, n); });
        report("float4_members", same_bytes(dout.download(), want), k, ref);
    }
    {
        auto a = make_vec<float2>(n, [&](int i) { return make_float2(f[2 * i] * 0.6f, f[2 * i + 1] * 0.6f); });
        std::vector<float2> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                float2 z = make_float2(0.f, 0.f), c = a[i];
                for (int k = 0; k < 8; ++k) {
                    z = cmul(z, z);
                    z.x += c.x, z.y += c.y;
                }
                want[i] = z;
            }
        });
        Buffer<float2> da(a), dout(n);
        double k = time_device([&] { k_float2<<<G, B>>>(da, dout, n); });
        // -a.y*b.y as an fma operand may be fused differently: tolerance.
        auto got = dout.download();
        bool ok = true;
        for (int i = 0; i < n && ok; ++i) {
            double tol = 1e-4 * (1 + std::fabs(want[i].x) + std::fabs(want[i].y));
            bool fin = std::isfinite(want[i].x) && std::isfinite(want[i].y);
            if (fin && (std::fabs(got[i].x - want[i].x) > tol || std::fabs(got[i].y - want[i].y) > tol))
                ok = false, std::printf("  mismatch %d: (%g,%g) want (%g,%g)\n", i, got[i].x, got[i].y, want[i].x, want[i].y);
        }
        report("float2_complex_iter", ok, k, ref);
    }
    {
        auto a = make_vec<int4>(n, [&](int i) { return make_int4(int(r[4 * i]), int(r[4 * i + 1]) >> 16, int(r[4 * i + 2]), int(r[4 * i + 3])); });
        auto b = make_vec<int2>(n, [&](int i) { return make_int2(int(r[i] >> 3), int(r[n + i]) >> 15); });
        std::vector<int4> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                int4 x = a[i];
                int2 y = b[i];
                want[i] = make_int4(int(unsigned(x.x) + unsigned(y.x)), x.y * y.y, x.z ^ int(unsigned(y.x) - unsigned(y.y)),
                                    int(unsigned(std::max(x.w, y.y)) - unsigned(std::min(x.x, y.x))));
            }
        });
        Buffer<int4> da(a), dout(n);
        Buffer<int2> db(b);
        double k = time_device([&] { k_int4<<<G, B>>>(da, db, dout, n); });
        report("int4_int2_minmax", same_bytes(dout.download(), want), k, ref);
    }
    {
        auto a = make_vec<double2>(n, [&](int i) { return make_double2(f[2 * i] * 3.0, f[2 * i + 1] * 1e-3); });
        std::vector<double2> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = d2_op(a[i], a[(i * 7 + 1) % n]);
        });
        Buffer<double2> da(a), dout(n);
        double k = time_device([&] { k_double2<<<G, B>>>(da, dout, n); });
        auto got = dout.download();
        bool ok = true;
        for (int i = 0; i < n && ok; ++i)
            if (std::fabs(got[i].x - want[i].x) > 1e-14 || std::fabs(got[i].y - want[i].y) > 1e-14) ok = false, std::printf("  mismatch %d\n", i);
        report("double2_complex", ok, k, ref);
    }
    {
        auto p = make_vec<uchar4>(n, [&](int i) { unsigned v = r[i]; return make_uchar4(v, v >> 8, v >> 16, v >> 24); });
        auto q = make_vec<uchar4>(n, [&](int i) { unsigned v = r[2 * n + i]; return make_uchar4(v, v >> 8, v >> 16, v >> 24); });
        std::vector<uchar4> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = blend(p[i], q[i]);
        });
        Buffer<uchar4> dp(p), dq(q), dout(n);
        double k = time_device([&] { k_uchar4<<<G, B>>>(dp, dq, dout, n); });
        // Rounding of x*(1-a)+y*a can differ by one step with fma contraction.
        auto got = dout.download();
        bool ok = true;
        for (int i = 0; i < n && ok; ++i)
            ok = std::abs(got[i].x - want[i].x) <= 1 && std::abs(got[i].y - want[i].y) <= 1 && std::abs(got[i].z - want[i].z) <= 1 &&
                 got[i].w == want[i].w;
        report("uchar4_blend", ok, k, ref);
    }
    {
        auto s = make_vec<short2>(n, [&](int i) { return make_short2(short(r[i]), short(r[i] >> 16)); });
        auto c = make_vec<char4>(n, [&](int i) { unsigned v = r[n + i]; return make_char4(v, v >> 8, v >> 16, v >> 24); });
        std::vector<int> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = small_op(s[i], c[i]);
        });
        Buffer<short2> ds(s);
        Buffer<char4> dc(c);
        Buffer<int> dout(n);
        double k = time_device([&] { k_small<<<G, B>>>(ds, dc, dout, n); });
        report("short2_char4", equal(dout.download().data(), want.data(), n), k, ref);
    }
    {
        auto u = make_vec<uint2>(n, [&](int i) { return make_uint2(r[i], r[n + i]); });
        auto l = make_vec<longlong2>(n, [&](int i) { return make_longlong2((long long)r[2 * n + i] << 20, -(long long)r[3 * n + i]); });
        std::vector<longlong2> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = wide_op(u[i], l[i]);
        });
        Buffer<uint2> du(u);
        Buffer<longlong2> dl(l), dout(n);
        double k = time_device([&] { k_wide<<<G, B>>>(du, dl, dout, n); });
        report("uint2_longlong2", same_bytes(dout.download(), want), k, ref);
    }
    {
        const int m = 4 * n + 3;
        std::vector<float> x(f.begin(), f.begin() + m), want(m);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int j = 0; j < m; ++j) want[j] = x[j] * float(2 + j % 4);
        });
        Buffer<float> dx(x), dy(m);
        int threads = m / 4 + 3;
        double k = time_device([&] { k_reinterpret<<<(threads + B - 1) / B, B>>>(dx, dy, m); });
        report("float_as_float4_with_tail", equal(dy.download().data(), want.data(), m), k, ref);
    }
    return finish();
}
