// Device function calls under divergence: lanes choose among large
// __noinline__ functions by data, noinline functions returning structs and
// taking many arguments, noinline chains, calls inside divergent loops,
// __forceinline__ helpers, template noinline functions.
#include "stress.cuh"

#ifdef __CUDACC__
#define NOINLINE __noinline__
#define FORCEINLINE __forceinline__
#else
#define NOINLINE
#define FORCEINLINE inline
#endif

__host__ __device__ FORCEINLINE unsigned rotl(unsigned x, int r) { return (x << r) | (x >> ((32 - r) & 31)); }
__host__ __device__ FORCEINLINE unsigned mixf(unsigned h, unsigned v) { return rotl(h ^ v, 13) * 5u + 0xe6546b64u; }

// Five large functions, each with its own loops and branches.
__host__ __device__ NOINLINE unsigned big0(unsigned x, int k) {
    unsigned h = x;
    for (int i = 0; i < 24 + k % 8; ++i) {
        h = mixf(h, unsigned(i));
        if (h & 0x10) h ^= h >> 7;
        else h += 0x1234567u;
    }
    return h;
}
__host__ __device__ NOINLINE unsigned big1(unsigned x, int k) {
    unsigned a = x, b = x ^ 0xdeadbeefu, c = unsigned(k);
    for (int i = 0; i < 16; ++i) {
        a -= c, a ^= rotl(c, 4), c += b;
        b -= a, b ^= rotl(a, 6), a += c;
        c -= b, c ^= rotl(b, 8), b += a;
        if ((a ^ b) & 1) c ^= 0x55aa55aau;
    }
    return a ^ b ^ c;
}
__host__ __device__ NOINLINE unsigned big2(unsigned x, int k) {
    float f = float(x & 0xffff) / 65536.0f, acc = 0;
    for (int i = 0; i < 20; ++i) {
        f = fmaf(f, 3.9f * (1.0f - f), 0.0f);  // logistic map
        if (f > 0.5f) acc += f;
        else acc -= 0.25f * f;
    }
    return unsigned(int(acc * 1000.0f)) ^ unsigned(k);
}
__host__ __device__ NOINLINE unsigned big3(unsigned x, int k) {
    unsigned long long a = x, m = 0x5851f42d4c957f2dull;
    for (int i = 0; i <= k % 30; ++i) {
        a = a * m + 1442695040888963407ull;
        switch ((a >> 60) & 3) {
            case 0: a ^= a >> 13; break;
            case 1: a += a << 7; break;
            case 2: a = ~a; break;
            default: a -= 12345; break;
        }
    }
    return unsigned(a ^ (a >> 32));
}
__host__ __device__ NOINLINE unsigned big4(unsigned x, int k) {
    unsigned digits = 0, v = x;
    while (v) {
        digits = digits * 7 + v % 10;
        v /= 10;
    }
    return digits + big0(x ^ digits, k + 1);  // a noinline function calling another
}
__global__ void k_dispatch(const unsigned* in, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = in[i];
    int k = int(x >> 24);
    unsigned r;
    switch (x % 5) {
        case 0: r = big0(x, k); break;
        case 1: r = big1(x, k); break;
        case 2: r = big2(x, k); break;
        case 3: r = big3(x, k); break;
        default: r = big4(x, k); break;
    }
    out[i] = r;
}
static unsigned dispatch_ref(unsigned x) {
    int k = int(x >> 24);
    switch (x % 5) {
        case 0: return big0(x, k);
        case 1: return big1(x, k);
        case 2: return big2(x, k);
        case 3: return big3(x, k);
        default: return big4(x, k);
    }
}

// Calls in a loop whose target changes per iteration and per lane.
__global__ void k_loop_calls(const unsigned* in, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned h = in[i];
    for (int it = 0; it < int(in[i] % 6); ++it) {
        if (h & 1) h = big1(h, it);
        else if (h & 2) h = big3(h, it);
        else h = big0(h, it) + 1;
    }
    out[i] = h;
}
static unsigned loop_calls_ref(unsigned x) {
    unsigned h = x;
    for (int it = 0; it < int(x % 6); ++it) {
        if (h & 1) h = big1(h, it);
        else if (h & 2) h = big3(h, it);
        else h = big0(h, it) + 1;
    }
    return h;
}

// Struct returned by value from a noinline function with many arguments.
struct Stats {
    int min, max;
    long long sum;
    float mean;
    unsigned char flags;
};
__host__ __device__ NOINLINE Stats summarize(int a, int b, int c, int d, int e, int f, int g, int h, short s, char ch, double w,
                                              float bias, bool neg) {
    int v[8] = {a, b, c, d, e, f, g, h};
    Stats st{v[0], v[0], 0, 0.0f, 0};
    for (int k = 0; k < 8; ++k) {
        int x = neg ? -v[k] : v[k];
        st.min = x < st.min ? x : st.min;
        st.max = x > st.max ? x : st.max;
        st.sum += x;
    }
    st.sum += s + ch;
    st.mean = float(st.sum) / 8.0f + bias;
    st.flags = (unsigned char)((st.sum > 0) | ((st.max - st.min > 1000) << 1) | ((w > 0.5) << 2));
    return st;
}
__global__ void k_struct_ret(const int* in, Stats* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int* p = in + 8 * i;
    out[i] = summarize(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], short(i), char(i & 0x7f), (i % 10) * 0.1, 0.25f, i & 1);
}

// Template noinline function instantiated for several widths.
template <class T, int Rounds>
__host__ __device__ NOINLINE T scramble(T x) {
    for (int r = 0; r < Rounds; ++r) {
        x ^= x >> (sizeof(T) * 2);
        x *= T(0x9E3779B97F4A7C15ull);
        if (x & 1) x += T(r);
    }
    return x;
}
__global__ void k_template_calls(const unsigned* in, unsigned long long* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = in[i];
    unsigned long long r;
    if (x & 1) r = scramble<unsigned long long, 7>(x);
    else if (x & 2) r = scramble<unsigned, 5>(x);
    else r = scramble<unsigned short, 9>((unsigned short)x);
    out[i] = r;
}
static unsigned long long template_ref(unsigned x) {
    if (x & 1) return scramble<unsigned long long, 7>(x);
    if (x & 2) return scramble<unsigned, 5>(x);
    return scramble<unsigned short, 9>((unsigned short)x);
}

int main() {
    using namespace stress;
    const int n = 1 << 17, B = 128, G = (n + B - 1) / B;
    auto r = random_u32(n, 1);
    Buffer<unsigned> din(r);
    {
        std::vector<unsigned> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = dispatch_ref(r[i]);
        });
        Buffer<unsigned> dout(n);
        double k = time_device([&] { k_dispatch<<<G, B>>>(din, dout, n); });
        report("divergent_noinline_dispatch", equal(dout.download().data(), want.data(), n), k, ref);
    }
    {
        std::vector<unsigned> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = loop_calls_ref(r[i]);
        });
        Buffer<unsigned> dout(n);
        double k = time_device([&] { k_loop_calls<<<G, B>>>(din, dout, n); });
        report("calls_in_divergent_loop", equal(dout.download().data(), want.data(), n), k, ref);
    }
    {
        std::vector<int> in(8 * size_t(n));
        auto v = random_u32(in.size(), 2);
        for (size_t k = 0; k < in.size(); ++k) in[k] = int(v[k] % 4001) - 2000;
        std::vector<Stats> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                const int* p = in.data() + 8 * i;
                want[i] = summarize(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], short(i), char(i & 0x7f), (i % 10) * 0.1, 0.25f, i & 1);
            }
        });
        Buffer<int> dinp(in);
        Buffer<Stats> dout(n);
        double k = time_device([&] { k_struct_ret<<<G, B>>>(dinp, dout, n); });
        auto got = dout.download();
        bool ok = true;
        for (int i = 0; i < n && ok; ++i)
            ok = got[i].min == want[i].min && got[i].max == want[i].max && got[i].sum == want[i].sum && got[i].mean == want[i].mean &&
                 got[i].flags == want[i].flags;
        report("noinline_13_args_struct_return", ok, k, ref);
    }
    {
        std::vector<unsigned long long> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = template_ref(r[i]);
        });
        Buffer<unsigned long long> dout(n);
        double k = time_device([&] { k_template_calls<<<G, B>>>(din, dout, n); });
        report("template_noinline_widths", equal(dout.download().data(), want.data(), n), k, ref);
    }
    return finish();
}
