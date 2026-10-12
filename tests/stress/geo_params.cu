// Kernel parameters: many mixed scalars, structs by value (padding, 256 B,
// arrays inside, 4000 B), arrays of device pointers, template kernels with type
// and int parameters, functors, device lambdas, constexpr and static members.
#include "stress.cuh"

enum class Mode : unsigned char { Add = 1, Mul = 2, Xor = 3 };

// 24 parameters of mixed types and sizes, in an order that forces padding.
__global__ void many_params(long long* out, char a, short b, int c, long long d, float e, double f, bool g, unsigned h,
                            unsigned char i, signed char j, unsigned short k, unsigned long long l, Mode m,
                            const int* p, float q, char r, double s, int t, short u, long long v, bool w, float x,
                            unsigned char y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    long long acc = a + b + c + d + (long long)e + (long long)f + g + h + i + j + k + (long long)l + (int)m;
    acc = acc * 3 + p[idx] + (long long)(q * 4) + r + (long long)s + t + u + v + w + (long long)(x * 8) + y;
    out[idx] = acc + idx;
}

struct Padded {  // float, char, double, int, char, short: with padding holes
    float f;
    char c;
    double d;
    int i;
    char c2;
    short s;
};
static_assert(sizeof(Padded) == 24, "layout");

__host__ __device__ inline double use_padded(const Padded& p, int i) {
    return p.f * i + p.c + p.d / (i + 1) + (p.i ^ i) + p.c2 * 2 + p.s;
}
__global__ void struct_by_value(Padded p, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = use_padded(p, i);
}

struct Big256 {
    int coef[48];
    double w[8];
};
static_assert(sizeof(Big256) == 256, "256 bytes");
__host__ __device__ inline double use_big(const Big256& b, int i) {
    double acc = 0;
    for (int k = 0; k < 48; ++k) acc += b.coef[(k + i) % 48] * (k + 1);
    return acc * b.w[i % 8];
}
__global__ void struct_256(Big256 b, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = use_big(b, i);
}

// Matrix inside a struct, applied to a float4 per thread.
struct Transform {
    float m[4][4];
    float bias[4];
    int rows;
};
__host__ __device__ inline float4 apply(const Transform& t, float4 v) {
    float in[4] = {v.x, v.y, v.z, v.w}, o[4];
    for (int r = 0; r < 4; ++r) {
        float a = t.bias[r];
        for (int c = 0; c < 4; ++c) a = fmaf(t.m[r][c], in[c], a);
        o[r] = r < t.rows ? a : in[r];
    }
    return make_float4(o[0], o[1], o[2], o[3]);
}
__global__ void struct_matrix(Transform t, const float4* in, float4* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = apply(t, in[i]);
}

// 4000-byte struct by value (close to the classic 4 KB parameter limit).
struct Huge {
    unsigned table[996];
    int n, shift, mask, pad;
};
static_assert(sizeof(Huge) == 4000, "4000 bytes");
__global__ void struct_4000(Huge h, unsigned* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < h.n) out[i] = h.table[(i * 7) % 996] ^ ((unsigned)i << h.shift & h.mask);
}

// Array of device pointers (pointer to pointer): row-wise jagged sums.
__global__ void jagged(const float* const* rows, const int* lens, float* sums, int nrows) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    const float* row = rows[r];
    float s = 0;
    for (int j = 0; j < lens[r]; ++j) s += row[j];
    sums[r] = s;
}

// Template kernel: type and int parameters.
template <class T, int N>
__global__ void tpl_window(const T* in, T* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    T acc = T(0);
#pragma unroll
    for (int k = 0; k < N; ++k) acc = acc + in[(i + k) % n] * T(k + 1);
    out[i] = acc;
}
template <class T, int N>
std::vector<T> tpl_window_ref(const std::vector<T>& in) {
    int n = int(in.size());
    std::vector<T> out(n);
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        T acc = T(0);
        for (int k = 0; k < N; ++k) acc = acc + in[(i + k) % n] * T(k + 1);
        out[i] = acc;
    }
    return out;
}

// Functors passed by value to a template kernel.
struct Saxpy {
    float a;
    __host__ __device__ float operator()(float x, float y) const { return fmaf(a, x, y); }
};
struct Clamp {
    float lo, hi;
    __host__ __device__ float operator()(float x, float y) const { return fminf(fmaxf(x + y, lo), hi); }
};
struct Polynomial {
    float c[6];
    __host__ __device__ float operator()(float x, float y) const {
        float r = c[5];
        for (int k = 4; k >= 0; --k) r = fmaf(r, x, c[k]);
        return r - y;
    }
};
template <class F>
__global__ void zip_with(F f, const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f(x[i], y[i]);
}

// Lambda defined in device code, passed to a device template function.
template <class F>
__device__ int reduce_digits(unsigned v, F f) {
    int acc = 0;
    while (v) {
        acc = f(acc, int(v % 10));
        v /= 10;
    }
    return acc;
}
__global__ void device_lambda(const unsigned* in, int* out, int n, int w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    auto weighted = [w](int acc, int d) { return acc * w + d; };
    auto sq = [](int acc, int d) { return acc + d * d; };
    out[i] = (i & 1) ? reduce_digits(in[i], weighted) : reduce_digits(in[i], sq);
}
static int reduce_digits_ref(unsigned v, int w, bool weighted) {
    int acc = 0;
    while (v) {
        int d = v % 10;
        acc = weighted ? acc * w + d : acc + d * d;
        v /= 10;
    }
    return acc;
}

// constexpr functions used for sizes, template arguments and at run time.
__host__ __device__ constexpr int ce_fact(int n) { return n <= 1 ? 1 : n * ce_fact(n - 1); }
__host__ __device__ constexpr unsigned ce_rotl(unsigned x, int r) { return (x << r) | (x >> (32 - r)); }
template <int K>
struct Table {
    static constexpr int size = ce_fact(K);
};
__global__ void constexpr_kernel(const unsigned* in, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    constexpr int sz = Table<5>::size;  // 120
    unsigned buf[sz % 7 + 3];          // 4 entries
    for (int k = 0; k < sz % 7 + 3; ++k) buf[k] = ce_rotl(in[i], k + 1) + ce_fact(k);
    unsigned acc = 0;
    for (int k = 0; k < sz % 7 + 3; ++k) acc ^= buf[k] * (k + 1);
    out[i] = acc + sz;
}
static unsigned constexpr_ref(unsigned x) {
    unsigned acc = 0;
    for (int k = 0; k < Table<5>::size % 7 + 3; ++k) acc ^= (ce_rotl(x, k + 1) + ce_fact(k)) * (k + 1);
    return acc + 120;
}

// Static member functions as a policy.
struct MinPolicy {
    __host__ __device__ static int init() { return 0x7fffffff; }
    __host__ __device__ static int op(int a, int b) { return a < b ? a : b; }
};
struct SumPolicy {
    __host__ __device__ static int init() { return 0; }
    __host__ __device__ static int op(int a, int b) { return a + b; }
};
template <class P>
__global__ void row_reduce(const int* in, int* out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    int acc = P::init();
    for (int c = 0; c < cols; ++c) acc = P::op(acc, in[r * cols + c]);
    out[r] = acc;
}

int main() {
    using namespace stress;
    const int n = 100000;

    {
        auto pr = random_u32(n, 1);
        std::vector<int> p(n);
        for (int i = 0; i < n; ++i) p[i] = int(pr[i] % 1000) - 500;
        char a = -5; short b = -300; int c = 123456; long long d = -9876543210LL; float e = 2.75f; double f = -1234.5;
        bool g = true; unsigned h = 4000000000u; unsigned char i8 = 250; signed char j = -100; unsigned short k = 65000;
        unsigned long long l = 1ull << 40; Mode m = Mode::Xor; float q = 1.25f; char r = 'z'; double s = 1e10;
        int t = -77; short u = 31000; long long v = 1LL << 50; bool w = false; float x = -0.5f; unsigned char y = 7;
        std::vector<long long> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int idx = 0; idx < n; ++idx) {
                long long acc = a + b + c + d + (long long)e + (long long)f + g + h + i8 + j + k + (long long)l + (int)m;
                acc = acc * 3 + p[idx] + (long long)(q * 4) + r + (long long)s + t + u + v + w + (long long)(x * 8) + y;
                want[idx] = acc + idx;
            }
        });
        Buffer<int> dp(p);
        Buffer<long long> dout(n);
        double kt = time_device([&] {
            many_params<<<(n + 127) / 128, 128>>>(dout, a, b, c, d, e, f, g, h, i8, j, k, l, m, dp, q, r, s, t, u, v, w, x,
                                                  y, n);
        });
        report("many_params_24", equal(dout.download().data(), want.data(), n), kt, ref);
    }

    {
        Padded p{3.5f, -7, 1e-3, 0x5a5a, 100, -1234};
        std::vector<double> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = use_padded(p, i);
        });
        Buffer<double> d(n);
        double k = time_device([&] { struct_by_value<<<(n + 255) / 256, 256>>>(p, d, n); });
        report("struct_padded_by_value", close(d.download().data(), want.data(), n, 1e-15), k, ref);
    }

    {
        Big256 b;
        for (int k = 0; k < 48; ++k) b.coef[k] = k * k - 300;
        for (int k = 0; k < 8; ++k) b.w[k] = 0.5 + k;
        std::vector<double> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = use_big(b, i);
        });
        Buffer<double> d(n);
        double k = time_device([&] { struct_256<<<(n + 99) / 100, 100>>>(b, d, n); });
        report("struct_256_bytes", equal(d.download().data(), want.data(), n), k, ref);
    }

    {
        Transform t;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) t.m[r][c] = (r == c ? 1.0f : 0.0f) + 0.125f * (r - c);
            t.bias[r] = 0.25f * r;
        }
        t.rows = 3;
        auto f = random_floats(4 * size_t(n), 3);
        std::vector<float4> in(n), want(n);
        for (int i = 0; i < n; ++i) in[i] = make_float4(f[4 * i], f[4 * i + 1], f[4 * i + 2], f[4 * i + 3]);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = apply(t, in[i]);
        });
        Buffer<float4> din(in), dout(n);
        double k = time_device([&] { struct_matrix<<<(n + 63) / 64, 64>>>(t, din, dout, n); });
        auto got = dout.download();
        bool ok = std::memcmp(got.data(), want.data(), n * sizeof(float4)) == 0;  // explicit fmaf: exact
        report("struct_array_member_matrix", ok, k, ref);
    }

    {
        Huge h;
        auto r = random_u32(996, 4);
        std::copy(r.begin(), r.end(), h.table);
        h.n = n;
        h.shift = 5;
        h.mask = 0x00ffff00;
        h.pad = 0;
        std::vector<unsigned> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = h.table[(i * 7) % 996] ^ ((unsigned)i << h.shift & h.mask);
        });
        Buffer<unsigned> d(n);
        double k = time_device([&] { struct_4000<<<(n + 255) / 256, 256>>>(h, d); });
        report("struct_4000_bytes", equal(d.download().data(), want.data(), n), k, ref);
    }

    {
        const int nrows = 3000;
        std::vector<int> lens(nrows);
        std::vector<std::vector<float>> rows(nrows);
        std::vector<float*> dptrs(nrows);
        for (int r = 0; r < nrows; ++r) {
            lens[r] = 1 + int(hash32(r) % 200);
            rows[r] = random_floats(lens[r], 100 + r);
            STRESS_CHECK(cudaMalloc(&dptrs[r], lens[r] * sizeof(float)));
            STRESS_CHECK(cudaMemcpy(dptrs[r], rows[r].data(), lens[r] * sizeof(float), cudaMemcpyHostToDevice));
        }
        std::vector<float> want(nrows);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int r = 0; r < nrows; ++r) {
                float s = 0;
                for (int j = 0; j < lens[r]; ++j) s += rows[r][j];
                want[r] = s;
            }
        });
        Buffer<float*> dp(dptrs);
        Buffer<int> dl(lens);
        Buffer<float> ds(nrows);
        double k = time_device([&] { jagged<<<(nrows + 127) / 128, 128>>>(dp, dl, ds, nrows); });
        report("pointer_to_pointer_jagged", equal(ds.download().data(), want.data(), nrows), k, ref);
        for (auto p : dptrs) cudaFree(p);
    }

    // Template instantiations.
    {
        auto r = random_u32(n, 7);
        std::vector<int> vi(n);
        std::vector<unsigned char> vb(n);
        std::vector<double> vd(n);
        std::vector<long long> vl(n);
        for (int i = 0; i < n; ++i) {
            vi[i] = int(r[i] % 20001) - 10000;
            vb[i] = (unsigned char)r[i];
            vd[i] = double(int(r[i] % 4096)) / 64.0;  // exact sums
            vl[i] = (long long)r[i] << 20;
        }
        auto run = [&](const char* name, auto& v, auto kernel, auto refv) {
            using T = typename std::decay_t<decltype(v)>::value_type;
            std::vector<T> want;
            double ref = best_ms([&] { want = refv(v); });
            Buffer<T> din(v), dout(v.size());
            double k = time_device([&] { kernel<<<(n + 255) / 256, 256>>>(din, dout, n); });
            report(name, equal(dout.download().data(), want.data(), v.size()), k, ref);
        };
        run("template_int_1", vi, tpl_window<int, 1>, tpl_window_ref<int, 1>);
        run("template_int_8", vi, tpl_window<int, 8>, tpl_window_ref<int, 8>);
        run("template_uchar_3", vb, tpl_window<unsigned char, 3>, tpl_window_ref<unsigned char, 3>);
        run("template_double_5", vd, tpl_window<double, 5>, tpl_window_ref<double, 5>);
        run("template_ll_16", vl, tpl_window<long long, 16>, tpl_window_ref<long long, 16>);
    }

    // Functors.
    {
        auto x = random_floats(n, 21), y = random_floats(n, 22);
        Buffer<float> dx(x), dy(y), dout(n);
        auto run = [&](const char* name, auto f) {
            std::vector<float> want(n);
            double ref = best_ms([&] {
#pragma omp parallel for
                for (int i = 0; i < n; ++i) want[i] = f(x[i], y[i]);
            });
            double k = time_device([&] { zip_with<<<(n + 255) / 256, 256>>>(f, dx, dy, dout, n); });
            report(name, equal(dout.download().data(), want.data(), n), k, ref);
        };
        run("functor_saxpy", Saxpy{2.5f});
        run("functor_clamp", Clamp{-0.5f, 0.75f});
        run("functor_polynomial", Polynomial{{1.0f, -0.5f, 0.25f, 0.125f, -0.0625f, 0.03125f}});
    }

    {
        auto in = random_u32(n, 8);
        std::vector<int> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = reduce_digits_ref(in[i], 3, i & 1);
        });
        Buffer<unsigned> din(in);
        Buffer<int> dout(n);
        double k = time_device([&] { device_lambda<<<(n + 255) / 256, 256>>>(din, dout, n, 3); });
        report("device_lambda_capture", equal(dout.download().data(), want.data(), n), k, ref);
    }

    {
        auto in = random_u32(n, 9);
        std::vector<unsigned> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = constexpr_ref(in[i]);
        });
        Buffer<unsigned> din(in), dout(n);
        double k = time_device([&] { constexpr_kernel<<<(n + 255) / 256, 256>>>(din, dout, n); });
        report("constexpr_functions", equal(dout.download().data(), want.data(), n), k, ref);
    }

    {
        const int rows = 20000, cols = 37;
        auto r = random_u32(size_t(rows) * cols, 10);
        std::vector<int> m(r.size());
        for (size_t i = 0; i < r.size(); ++i) m[i] = int(r[i] % 100000) - 50000;
        std::vector<int> wmin(rows), wsum(rows);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < rows; ++i) {
                int mn = MinPolicy::init(), s = SumPolicy::init();
                for (int c = 0; c < cols; ++c) {
                    mn = MinPolicy::op(mn, m[i * cols + c]);
                    s = SumPolicy::op(s, m[i * cols + c]);
                }
                wmin[i] = mn;
                wsum[i] = s;
            }
        });
        Buffer<int> dm(m), dmin(rows), dsum(rows);
        double k = time_device([&] {
            row_reduce<MinPolicy><<<(rows + 255) / 256, 256>>>(dm, dmin, rows, cols);
            row_reduce<SumPolicy><<<(rows + 255) / 256, 256>>>(dm, dsum, rows, cols);
        });
        report("static_member_policies",
               equal(dmin.download().data(), wmin.data(), rows) && equal(dsum.download().data(), wsum.data(), rows), k, ref);
    }
    return finish();
}
