// Double precision heavy kernels. All products feeding a sum are written
// with explicit fma() so device and host do the same rounding (nvcc would
// otherwise contract a*b+c on the device only): results are compared exactly.
#include "stress.cuh"

__host__ __device__ inline double horner(double x, const double* c, int deg) {
    double r = c[deg];
    for (int k = deg - 1; k >= 0; --k) r = fma(r, x, c[k]);
    return r;
}
struct Coefs {
    double c[21];
};
__global__ void k_horner(Coefs cf, const double* x, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = horner(x[i], cf.c, 20);
}

// Newton-Raphson reciprocal from a crude start: fma-only iterations.
__host__ __device__ inline double newton_recip(double a) {
    double x = 1.0 / double(float(a)) ;  // start
    for (int k = 0; k < 4; ++k) {
        double e = fma(-a, x, 1.0);
        x = fma(x, e, x);
    }
    return x;
}
__global__ void k_newton(const double* a, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = newton_recip(a[i]);
}

// IEEE division and square root (correctly rounded on both sides).
__global__ void k_divsqrt(const double* a, const double* b, double* q, double* s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    q[i] = a[i] / b[i];
    s[i] = sqrt(fabs(a[i]));
}

// Kahan compensated summation of 256 terms per thread (adds only).
__host__ __device__ inline double kahan(const double* x, int start, int n) {
    double sum = 0, c = 0;
    for (int k = 0; k < 256; ++k) {
        double y = x[(start + k * 13) % n] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}
__global__ void k_kahan(const double* x, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = kahan(x, i, n);
}

// Gravity: acceleration on each body from all others, softened.
struct Body {
    double x, y, z, m;
};
__host__ __device__ inline void accel(const Body* b, int nb, int i, double* out) {
    double ax = 0, ay = 0, az = 0;
    Body p = b[i];
    for (int j = 0; j < nb; ++j) {
        double dx = b[j].x - p.x, dy = b[j].y - p.y, dz = b[j].z - p.z;
        double r2 = fma(dz, dz, fma(dy, dy, fma(dx, dx, 1e-9)));
        double inv = 1.0 / sqrt(r2);
        double s = b[j].m * (inv * inv * inv);
        ax = fma(dx, s, ax), ay = fma(dy, s, ay), az = fma(dz, s, az);
    }
    out[0] = ax, out[1] = ay, out[2] = az;
}
__global__ void k_nbody(const Body* b, double* acc, int nb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nb) accel(b, nb, i, acc + 3 * i);
}

// Conversions between double and integers/float.
__host__ __device__ inline long long conv(double d) {
    long long a = (long long)d;                       // truncation
    unsigned u = (unsigned)fabs(fmod(d, 4.0e9));      // in range
    float f = (float)d;                               // round to nearest
    double back = (double)a + (double)u * 0.5;        // exact for |a| < 2^53
    long long bits = (long long)(f * 2.0f);
    return a ^ (long long)u ^ (bits << 1) ^ (long long)(back * 3.0);
}
__global__ void k_conv(const double* d, long long* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = conv(d[i]);
}

// Mandelbrot iteration counts in double.
__host__ __device__ inline int mandel(double cx, double cy, int maxit) {
    double x = 0, y = 0;
    int it = 0;
    while (it < maxit && fma(x, x, y * y) <= 4.0) {
        double xt = fma(x, x, fma(-y, y, cx));
        y = fma(2.0 * x, y, cy);
        x = xt;
        ++it;
    }
    return it;
}
__global__ void k_mandel(int* out, int w, int h, int maxit) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;
    out[py * w + px] = mandel(-2.0 + 2.6 * px / w, -1.2 + 2.4 * py / h, maxit);
}

int main() {
    using namespace stress;
    const int n = 1 << 18, B = 256, G = (n + B - 1) / B;
    auto rf = random_floats(4 * size_t(n), 1);
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) x[i] = double(rf[i]) + double(rf[n + i]) * 1e-8;

    {
        Coefs cf;
        for (int k = 0; k <= 20; ++k) cf.c[k] = 1.0 / (k + 1) * (k % 3 == 0 ? -1 : 1);
        std::vector<double> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = horner(x[i], cf.c, 20);
        });
        Buffer<double> dx(x), dy(n);
        double k = time_device([&] { k_horner<<<G, B>>>(cf, dx, dy, n); });
        report("double_horner_fma_deg20", equal(dy.download().data(), want.data(), n), k, ref);
    }
    {
        std::vector<double> a(n), want(n);
        for (int i = 0; i < n; ++i) a[i] = (x[i] + 1.5) * std::ldexp(1.0, int(i % 40) - 20);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = newton_recip(a[i]);
        });
        Buffer<double> da(a), dout(n);
        double k = time_device([&] { k_newton<<<G, B>>>(da, dout, n); });
        report("double_newton_reciprocal", equal(dout.download().data(), want.data(), n), k, ref);
    }
    {
        std::vector<double> a(n), b(n), wq(n), ws(n);
        for (int i = 0; i < n; ++i) {
            a[i] = x[i] * std::ldexp(1.0, int(i % 200) - 100);
            b[i] = double(rf[2 * n + i]) * std::ldexp(1.0, int(i % 61) - 30);
            if (i % 101 == 0) b[i] = 0.0;
            if (i % 103 == 0) a[i] = 4.9e-324 * (i % 7);  // denormals
        }
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) wq[i] = a[i] / b[i], ws[i] = std::sqrt(std::fabs(a[i]));
        });
        Buffer<double> da(a), db(b), dq(n), ds(n);
        double k = time_device([&] { k_divsqrt<<<G, B>>>(da, db, dq, ds, n); });
        report("double_ieee_div_sqrt", close(dq.download().data(), wq.data(), n, 0.0, 0.0, "q") && equal(ds.download().data(), ws.data(), n, "s"), k, ref);
    }
    {
        std::vector<double> v(n), want(n);
        for (int i = 0; i < n; ++i) v[i] = x[i] * std::ldexp(1.0, int(i % 50) - 25);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = kahan(v.data(), i, n);
        });
        Buffer<double> dv(v), dout(n);
        double k = time_device([&] { k_kahan<<<G, B>>>(dv, dout, n); });
        report("double_kahan_sum_256", equal(dout.download().data(), want.data(), n), k, ref);
    }
    {
        const int nb = 2048;
        std::vector<Body> b(nb);
        for (int i = 0; i < nb; ++i) b[i] = Body{rf[4 * i] * 10.0, rf[4 * i + 1] * 10.0, rf[4 * i + 2] * 10.0, 1.0 + rf[4 * i + 3] * 0.5};
        std::vector<double> want(3 * nb);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < nb; ++i) accel(b.data(), nb, i, &want[3 * i]);
        });
        Buffer<Body> db(b);
        Buffer<double> dacc(3 * nb);
        double k = time_device([&] { k_nbody<<<nb / 128, 128>>>(db, dacc, nb); });
        report("double_nbody_2048", equal(dacc.download().data(), want.data(), 3 * nb), k, ref);
    }
    {
        std::vector<double> d(n);
        std::vector<long long> want(n);
        for (int i = 0; i < n; ++i) d[i] = x[i] * std::ldexp(1.0, int(i % 62)) + (i % 3 == 0 ? 0.5 : 0.0);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = conv(d[i]);
        });
        Buffer<double> dd(d);
        Buffer<long long> dout(n);
        double k = time_device([&] { k_conv<<<G, B>>>(dd, dout, n); });
        report("double_int_float_conversions", equal(dout.download().data(), want.data(), n), k, ref);
    }
    {
        const int w = 768, h = 512, maxit = 256;
        std::vector<int> want(w * h);
        double ref = best_ms([&] {
#pragma omp parallel for schedule(dynamic)
            for (int py = 0; py < h; ++py)
                for (int px = 0; px < w; ++px) want[py * w + px] = mandel(-2.0 + 2.6 * px / w, -1.2 + 2.4 * py / h, maxit);
        });
        Buffer<int> dout(w * h);
        double k = time_device([&] { k_mandel<<<dim3(w / 32, h / 8), dim3(32, 8)>>>(dout, w, h, maxit); });
        report("double_mandelbrot", equal(dout.download().data(), want.data(), w * h), k, ref);
    }
    return finish();
}
