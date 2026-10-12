// Integer types: 8/16/64-bit arithmetic, unsigned wrap-around, signed 64-bit
// division and modulo with negative operands, 128-bit products through
// __umul64hi/__mul64hi, bool arrays, enums, integer promotions.
#include "stress.cuh"

// 8-bit: char, signed char, unsigned char, with promotion and truncation.
__host__ __device__ inline int ops8(signed char a, signed char b, unsigned char ua, unsigned char ub, char c) {
    signed char s1 = (signed char)(a + b);
    signed char s2 = (signed char)(a * b);
    unsigned char u1 = (unsigned char)(ua + ub);
    unsigned char u2 = (unsigned char)(ua * ub);
    unsigned char u3 = (unsigned char)(ua - ub);
    int q = b != 0 ? a / b : 0;
    int r = b != 0 ? a % b : 0;
    int sh = (a >> 2) ^ (ua >> 3) ^ (unsigned char)(ub << 3);
    int cmp = (a < b) + 2 * (ua < ub) + 4 * (c == 'q') + 8 * (a < ua);
    char cc = char(c + 1);
    return s1 * 7 + s2 * 131 + u1 * 3 + u2 * 5 + u3 * 11 + q * 13 + r * 17 + sh * 19 + cmp * 23 + cc;
}
__global__ void k_ops8(const unsigned* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned v = in[i];
    out[i] = ops8((signed char)v, (signed char)(v >> 8), (unsigned char)(v >> 16), (unsigned char)(v >> 24), char(v ^ (v >> 13)));
}

// 16-bit.
__host__ __device__ inline int ops16(short a, short b, unsigned short ua, unsigned short ub) {
    short s1 = short(a + b), s2 = short(a * b), s3 = short(a - b);
    unsigned short u1 = (unsigned short)(ua + ub), u2 = (unsigned short)(ua * ub), u3 = (unsigned short)(ua - ub);
    int q = b ? a / b : -1, r = b ? a % b : -1;
    unsigned uq = ub ? ua / ub : 7u, ur = ub ? ua % ub : 7u;
    int sh = (a >> 5) + (ua >> 7) + short(a << 3);
    return s1 + 3 * s2 + 5 * s3 + 7 * u1 + 11 * u2 + 13 * u3 + 17 * q + 19 * r + 23 * int(uq) + 29 * int(ur) + 31 * sh;
}
__global__ void k_ops16(const unsigned* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned v = in[i], w = in[(i + 1) % n];
    out[i] = ops16(short(v), short(v >> 16), (unsigned short)w, (unsigned short)(w >> 16));
}

// 64-bit unsigned wrap-around and shifts.
__host__ __device__ inline unsigned long long ops_u64(unsigned long long a, unsigned long long b) {
    unsigned s = unsigned(b & 63);
    unsigned long long r = a * b;
    r ^= a + b;           // wraps
    r += a - b;           // wraps below zero
    r ^= (~a) << s;
    r += b >> s;
    r ^= (a * 0x9E3779B97F4A7C15ull) >> 7;
    r += (unsigned long long)(unsigned)(a) * (unsigned)(b >> 32);  // 32x32 -> 64
    r ^= (unsigned long long)(unsigned)(a + b);                    // truncation to 32 bits
    return r;
}
__global__ void k_u64(const unsigned long long* a, const unsigned long long* b, unsigned long long* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ops_u64(a[i], b[i]);
}

// 64-bit signed division and modulo, negative operands (truncation toward zero).
__host__ __device__ inline long long divmod64(long long a, long long b) {
    long long q = a / b, r = a % b;
    long long q2 = a / 7, r2 = a % -7;
    long long q3 = (a >> 20) / (b | 1), r3 = (a >> 20) % (b | 1);
    return q * 31 + r * 17 + q2 * 3 + r2 + q3 * 5 + r3 * 11;
}
__global__ void k_divmod64(const long long* a, const long long* b, long long* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = divmod64(a[i], b[i]);
}

// 32-bit unsigned division and modulo with the same operands (different path from 64).
__host__ __device__ inline unsigned divmod32(unsigned a, unsigned b) {
    unsigned d = b ? b : 1;
    int sa = int(a), sb = int(d | 1);
    return a / d + a % d * 3 + unsigned(sa / sb) * 5 + unsigned(sa % sb) * 7 + a / 1000003u + a % 641u;
}
__global__ void k_divmod32(const unsigned* a, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = divmod32(a[i], a[(i + 3) % n] >> (i % 32));
}

// 128-bit accumulate of 64x64 products: lo by multiplication, hi by __umul64hi.
struct U128 {
    unsigned long long lo, hi;
    bool operator==(const U128& o) const { return lo == o.lo && hi == o.hi; }
    explicit operator double() const { return double(hi) * 18446744073709551616.0 + double(lo); }
};
__global__ void k_mul128(const unsigned long long* a, const unsigned long long* b, U128* out, long long* shi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long lo = 0, hi = 0;
    for (int k = 0; k < 8; ++k) {
        unsigned long long x = a[(i + k) % n], y = b[(i * 3 + k) % n];
        unsigned long long plo = x * y, phi = __umul64hi(x, y);
        unsigned long long nlo = lo + plo;
        hi += phi + (nlo < lo);
        lo = nlo;
    }
    out[i] = U128{lo, hi};
    shi[i] = __mul64hi((long long)a[i], (long long)b[i]);
}

// bool arrays: a sieve-like marking and logical combinations.
__global__ void k_bool(const bool* a, const bool* b, bool* c, int* count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool x = a[i], y = b[i], z = a[(i + 1) % n];
    c[i] = (x && !y) || (y != z);
    int cnt = 0;
    for (int k = 0; k < 16; ++k) cnt += a[(i + k) % n];  // bool to int promotion
    count[i] = cnt + (x ^ y) * 100 + (bool(i & 4) == y);
}

// Enums: unscoped with explicit values, scoped with a small underlying type.
enum Color { Red = 3, Green = 7, Blue = -2 };
enum class Op : signed char { Inc = -1, Dbl = 0, Neg = 1, Sq = 2 };
__host__ __device__ inline int apply_enum(Op op, Color c, int v) {
    switch (op) {
        case Op::Inc: v += c; break;
        case Op::Dbl: v *= 2; break;
        case Op::Neg: v = -v + (c == Blue); break;
        case Op::Sq: v = v * (v & 0xff); break;
    }
    return v + int(op) * 1000 + int(sizeof(Op)) * 10;
}
__global__ void k_enum(const unsigned* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned v = in[i];
    Op op = Op(int(v % 4) - 1);
    Color c = (v >> 4) % 3 == 0 ? Red : (v >> 4) % 3 == 1 ? Green : Blue;
    out[i] = apply_enum(op, c, int(v >> 12));
}

// Usual arithmetic conversions between signed and unsigned.
__host__ __device__ inline long long promo(int a, unsigned b, signed char c, unsigned char d, long long e) {
    long long r = 0;
    r += (a < b);                      // a converted to unsigned
    r += (c < d) * 2;                  // both promoted to int
    r += (unsigned long long)c >> 60;  // sign extension then logical shift
    r += (long long)(unsigned)a;       // zero extension
    r += (long long)a;                 // sign extension
    r += (unsigned)c;                  // sign extended to 32, then unsigned
    r += e + b;                        // unsigned to long long
    r += (int)(unsigned short)a;
    r ^= (long long)(short)b;
    return r;
}
__global__ void k_promo(const unsigned* in, long long* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned v = in[i], w = in[(i + 7) % n];
    out[i] = promo(int(v), w, (signed char)(v >> 5), (unsigned char)(w >> 9), (long long)(int)w * -12345);
}

int main() {
    using namespace stress;
    const int n = 1 << 18;
    auto r32 = random_u32(n, 42);
    const int B = 256, G = (n + B - 1) / B;

    {
        std::vector<int> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                unsigned v = r32[i];
                want[i] = ops8((signed char)v, (signed char)(v >> 8), (unsigned char)(v >> 16), (unsigned char)(v >> 24), char(v ^ (v >> 13)));
            }
        });
        Buffer<unsigned> din(r32);
        Buffer<int> dout(n);
        double k = time_device([&] { k_ops8<<<G, B>>>(din, dout, n); });
        report("int8_char_schar_uchar", equal(dout.download().data(), want.data(), n), k, ref);
    }
    {
        std::vector<int> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                unsigned v = r32[i], w = r32[(i + 1) % n];
                want[i] = ops16(short(v), short(v >> 16), (unsigned short)w, (unsigned short)(w >> 16));
            }
        });
        Buffer<unsigned> din(r32);
        Buffer<int> dout(n);
        double k = time_device([&] { k_ops16<<<G, B>>>(din, dout, n); });
        report("int16_short_ushort", equal(dout.download().data(), want.data(), n), k, ref);
    }

    std::vector<unsigned long long> a64(n), b64(n);
    auto r2 = random_u32(n, 43), r3 = random_u32(n, 44);
    for (int i = 0; i < n; ++i) {
        a64[i] = (unsigned long long)r32[i] << 32 | r2[i];
        b64[i] = (unsigned long long)r3[i] << 32 | r32[(i + 5) % n];
        if (i % 17 == 0) a64[i] = ~0ull - i;  // near the top of the range
        if (i % 19 == 0) b64[i] = i;          // small
    }
    {
        std::vector<unsigned long long> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = ops_u64(a64[i], b64[i]);
        });
        Buffer<unsigned long long> da(a64), db(b64), dout(n);
        double k = time_device([&] { k_u64<<<G, B>>>(da, db, dout, n); });
        report("uint64_wraparound", equal(dout.download().data(), want.data(), n), k, ref);
    }
    {
        std::vector<long long> sa(n), sb(n), want(n);
        for (int i = 0; i < n; ++i) {
            sa[i] = (long long)a64[i];
            long long d = (long long)b64[i] >> (r2[i] % 60);
            sb[i] = d == 0 ? -3 : d;
            if (i % 5 == 0) sb[i] = -(long long)(r3[i] % 1000) - 1;  // small negative divisors
            if (i % 7 == 0) sa[i] = -(long long)(r2[i] % 100);       // small negative dividends
        }
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = divmod64(sa[i], sb[i]);
        });
        Buffer<long long> da(sa), db(sb), dout(n);
        double k = time_device([&] { k_divmod64<<<G, B>>>(da, db, dout, n); });
        report("int64_divmod_negative", equal(dout.download().data(), want.data(), n), k, ref);
    }
    {
        std::vector<unsigned> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = divmod32(r32[i], r32[(i + 3) % n] >> (i % 32));
        });
        Buffer<unsigned> din(r32), dout(n);
        double k = time_device([&] { k_divmod32<<<G, B>>>(din, dout, n); });
        report("int32_divmod_mixed", equal(dout.download().data(), want.data(), n), k, ref);
    }
    {
        std::vector<U128> want(n);
        std::vector<long long> whi(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                unsigned __int128 acc = 0;
                for (int k = 0; k < 8; ++k) acc += (unsigned __int128)a64[(i + k) % n] * b64[(i * 3 + k) % n];
                want[i] = U128{(unsigned long long)acc, (unsigned long long)(acc >> 64)};
                whi[i] = (long long)(((__int128)(long long)a64[i] * (long long)b64[i]) >> 64);
            }
        });
        Buffer<unsigned long long> da(a64), db(b64);
        Buffer<U128> dout(n);
        Buffer<long long> dhi(n);
        double k = time_device([&] { k_mul128<<<G, B>>>(da, db, dout, dhi, n); });
        auto got = dout.download();
        bool ok = std::equal(got.begin(), got.end(), want.begin()) && equal(dhi.download().data(), whi.data(), n, "mul64hi");
        report("u128_via_umul64hi", ok, k, ref);
    }
    {
        std::vector<char> a(n), b(n);  // std::vector<bool> has no data()
        for (int i = 0; i < n; ++i) a[i] = r32[i] & 1, b[i] = (r32[i] >> 7) % 3 == 0;
        std::vector<char> wc(n);
        std::vector<int> wcount(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                bool x = a[i], y = b[i], z = a[(i + 1) % n];
                wc[i] = (x && !y) || (y != z);
                int cnt = 0;
                for (int k = 0; k < 16; ++k) cnt += (bool)a[(i + k) % n];
                wcount[i] = cnt + (x ^ y) * 100 + (bool(i & 4) == y);
            }
        });
        bool *da, *db, *dc;
        STRESS_CHECK(cudaMalloc(&da, n));
        STRESS_CHECK(cudaMalloc(&db, n));
        STRESS_CHECK(cudaMalloc(&dc, n));
        STRESS_CHECK(cudaMemcpy(da, a.data(), n, cudaMemcpyHostToDevice));
        STRESS_CHECK(cudaMemcpy(db, b.data(), n, cudaMemcpyHostToDevice));
        Buffer<int> dcount(n);
        double k = time_device([&] { k_bool<<<G, B>>>(da, db, dc, dcount, n); });
        std::vector<char> gc(n);
        STRESS_CHECK(cudaMemcpy(gc.data(), dc, n, cudaMemcpyDeviceToHost));
        bool ok = equal(gc.data(), wc.data(), n, "c") && equal(dcount.download().data(), wcount.data(), n, "count");
        report("bool_arrays", ok, k, ref);
        cudaFree(da), cudaFree(db), cudaFree(dc);
    }
    {
        std::vector<int> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                unsigned v = r32[i];
                Op op = Op(int(v % 4) - 1);
                Color c = (v >> 4) % 3 == 0 ? Red : (v >> 4) % 3 == 1 ? Green : Blue;
                want[i] = apply_enum(op, c, int(v >> 12));
            }
        });
        Buffer<unsigned> din(r32);
        Buffer<int> dout(n);
        double k = time_device([&] { k_enum<<<G, B>>>(din, dout, n); });
        report("enums_scoped_unscoped", equal(dout.download().data(), want.data(), n), k, ref);
    }
    {
        std::vector<long long> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                unsigned v = r32[i], w = r32[(i + 7) % n];
                want[i] = promo(int(v), w, (signed char)(v >> 5), (unsigned char)(w >> 9), (long long)(int)w * -12345);
            }
        });
        Buffer<unsigned> din(r32);
        Buffer<long long> dout(n);
        double k = time_device([&] { k_promo<<<G, B>>>(din, dout, n); });
        report("signed_unsigned_promotions", equal(dout.download().data(), want.data(), n), k, ref);
    }
    return finish();
}
