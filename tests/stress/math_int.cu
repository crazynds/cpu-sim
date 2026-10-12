// Integer intrinsics, one case per intrinsic: bit counting, bit reversal,
// byte permutes, funnel shifts, 24-bit and high multiplies, sums of absolute
// differences, dot products of packed bytes, SIMD-in-a-word byte ops,
// halving adds, min/max/abs. Host references are written from the
// definitions in the CUDA Math API documentation, independently of the
// intrinsics.
#include "stress.cuh"

using u32 = unsigned;
using u64 = unsigned long long;
using i64 = long long;

enum Fn {
    POPC, POPCLL, CLZ, CLZLL, FFS, FFSLL, BREV, BREVLL, BYTE_PERM, FSHL, FSHR, FSHL_C, FSHR_C, MUL24, UMUL24, MULHI, UMULHI,
    MUL64HI, UMUL64HI, SAD, USAD, DP4A, DP4A_U, DP2A_LO, DP2A_HI, VABSDIFF4, VADD4, VSUB4, HADD, RHADD, UHADD, URHADD,
    MINMAX_I, MINMAX_U, MINMAX_LL, ABS_I, ABS_LL, NFN
};
static const char* names[NFN] = {
    "popc", "popcll", "clz", "clzll", "ffs", "ffsll", "brev", "brevll", "byte_perm", "funnelshift_l", "funnelshift_r",
    "funnelshift_lc", "funnelshift_rc", "mul24", "umul24", "mulhi", "umulhi", "mul64hi", "umul64hi", "sad", "usad", "dp4a",
    "dp4a_unsigned", "dp2a_lo", "dp2a_hi", "vabsdiffu4", "vadd4", "vsub4", "hadd", "rhadd", "uhadd", "urhadd", "min_max_int",
    "min_max_unsigned", "min_max_longlong", "abs_int", "llabs"};

template <int F>
__device__ u64 dev_op(u32 a, u32 b, u32 c, u64 A, u64 B) {
    int sa = int(a), sb = int(b);
    if constexpr (F == POPC) return __popc(a);
    else if constexpr (F == POPCLL) return __popcll(A);
    else if constexpr (F == CLZ) return __clz(sa);
    else if constexpr (F == CLZLL) return __clzll(i64(A));
    else if constexpr (F == FFS) return __ffs(sa);
    else if constexpr (F == FFSLL) return __ffsll(i64(A));
    else if constexpr (F == BREV) return __brev(a);
    else if constexpr (F == BREVLL) return __brevll(A);
    else if constexpr (F == BYTE_PERM) return __byte_perm(a, b, c & 0x7777);
    else if constexpr (F == FSHL) return __funnelshift_l(a, b, c);
    else if constexpr (F == FSHR) return __funnelshift_r(a, b, c);
    else if constexpr (F == FSHL_C) return __funnelshift_lc(a, b, c % 40);
    else if constexpr (F == FSHR_C) return __funnelshift_rc(a, b, c % 40);
    else if constexpr (F == MUL24) return u32(__mul24(sa, sb));
    else if constexpr (F == UMUL24) return __umul24(a, b);
    else if constexpr (F == MULHI) return u32(__mulhi(sa, sb));
    else if constexpr (F == UMULHI) return __umulhi(a, b);
    else if constexpr (F == MUL64HI) return u64(__mul64hi(i64(A), i64(B)));
    else if constexpr (F == UMUL64HI) return __umul64hi(A, B);
    else if constexpr (F == SAD) return __sad(sa, sb, c);
    else if constexpr (F == USAD) return __usad(a, b, c);
    else if constexpr (F == DP4A) return u32(__dp4a(sa, sb, int(c)));
    else if constexpr (F == DP4A_U) return __dp4a(a, b, c);
    else if constexpr (F == DP2A_LO) return u32(__dp2a_lo(sa, sb, int(c)));
    else if constexpr (F == DP2A_HI) return u32(__dp2a_hi(sa, sb, int(c)));
    else if constexpr (F == VABSDIFF4) return __vabsdiffu4(a, b);
    else if constexpr (F == VADD4) return __vadd4(a, b);
    else if constexpr (F == VSUB4) return __vsub4(a, b);
    else if constexpr (F == HADD) return u32(__hadd(sa, sb));
    else if constexpr (F == RHADD) return u32(__rhadd(sa, sb));
    else if constexpr (F == UHADD) return __uhadd(a, b);
    else if constexpr (F == URHADD) return __urhadd(a, b);
    else if constexpr (F == MINMAX_I) return u64(u32(min(sa, sb))) << 32 | u32(max(sa, int(c)));
    else if constexpr (F == MINMAX_U) return u64(min(a, b)) << 32 | max(b, c);
    else if constexpr (F == MINMAX_LL) return u64(min(i64(A), i64(B))) ^ (u64(max(i64(A), i64(B))) << 1);
    else if constexpr (F == ABS_I) return u32(abs(sa == INT32_MIN ? 0 : sa));
    else return u64(llabs(i64(A) == INT64_MIN ? 1 : i64(A)));
}

template <int F>
__global__ void k_op(const u32* a, const u32* b, const u32* c, const u64* A, const u64* B, u64* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = dev_op<F>(a[i], b[i], c[i], A[i], B[i]);
}

// Host references from the definitions.
static int popc64(u64 x) {
    int c = 0;
    for (; x; x &= x - 1) ++c;
    return c;
}
static int clz64(u64 x, int bits) {
    int c = 0;
    for (int k = bits - 1; k >= 0 && !((x >> k) & 1); --k) ++c;
    return c;
}
static int ffs64(u64 x, int bits) {
    for (int k = 0; k < bits; ++k)
        if ((x >> k) & 1) return k + 1;
    return 0;
}
static u64 brev64(u64 x, int bits) {
    u64 r = 0;
    for (int k = 0; k < bits; ++k) r |= ((x >> k) & 1) << (bits - 1 - k);
    return r;
}
static int sbyte(u32 x, int k) { return int((signed char)(x >> (8 * k))); }
static int ubyte(u32 x, int k) { return int((x >> (8 * k)) & 0xff); }
static int sx24(u32 x) { return int(x << 8) >> 8; }

static u64 ref_op(int f, u32 a, u32 b, u32 c, u64 A, u64 B) {
    int sa = int(a), sb = int(b);
    switch (f) {
        case POPC: return popc64(a);
        case POPCLL: return popc64(A);
        case CLZ: return clz64(a, 32);
        case CLZLL: return clz64(A, 64);
        case FFS: return ffs64(a, 32);
        case FFSLL: return ffs64(A, 64);
        case BREV: return brev64(a, 32);
        case BREVLL: return brev64(A, 64);
        case BYTE_PERM: {
            u64 in = u64(b) << 32 | a;
            u32 s = c & 0x7777, r = 0;
            for (int k = 0; k < 4; ++k) r |= u32((in >> (8 * ((s >> (4 * k)) & 7))) & 0xff) << (8 * k);
            return r;
        }
        case FSHL: return u32((((u64(b) << 32) | a) << (c & 31)) >> 32);
        case FSHR: return u32(((u64(b) << 32) | a) >> (c & 31));
        case FSHL_C: {
            u32 s = std::min(c % 40, 32u);
            return s == 32 ? a : u32((((u64(b) << 32) | a) << s) >> 32);
        }
        case FSHR_C: {
            u32 s = std::min(c % 40, 32u);
            return s == 32 ? b : u32(((u64(b) << 32) | a) >> s);
        }
        case MUL24: return u32(i64(sx24(a)) * sx24(b));
        case UMUL24: return u32(u64(a & 0xffffff) * (b & 0xffffff));
        case MULHI: return u32(u64((i64(sa) * sb)) >> 32);
        case UMULHI: return u32((u64(a) * b) >> 32);
        case MUL64HI: return u64(i64(((__int128)i64(A) * i64(B)) >> 64));
        case UMUL64HI: return u64(((unsigned __int128)A * B) >> 64);
        case SAD: return u32(std::llabs(i64(sa) - sb) + c);
        case USAD: return u32((a > b ? a - b : b - a) + c);
        case DP4A: {
            int s = int(c);
            for (int k = 0; k < 4; ++k) s = int(u32(s) + u32(sbyte(a, k) * sbyte(b, k)));
            return u32(s);
        }
        case DP4A_U: {
            u32 s = c;
            for (int k = 0; k < 4; ++k) s += u32(ubyte(a, k) * ubyte(b, k));
            return s;
        }
        case DP2A_LO:
        case DP2A_HI: {
            int k0 = f == DP2A_LO ? 0 : 2;
            int a0 = int(short(a)), a1 = int(short(a >> 16));
            return u32(u32(c) + u32(a0 * sbyte(b, k0)) + u32(a1 * sbyte(b, k0 + 1)));
        }
        case VABSDIFF4:
        case VADD4:
        case VSUB4: {
            u32 r = 0;
            for (int k = 0; k < 4; ++k) {
                int x = ubyte(a, k), y = ubyte(b, k);
                int v = f == VABSDIFF4 ? std::abs(x - y) : f == VADD4 ? x + y : x - y;
                r |= u32(v & 0xff) << (8 * k);
            }
            return r;
        }
        case HADD: return u32(int((i64(sa) + sb) >> 1));
        case RHADD: return u32(int((i64(sa) + sb + 1) >> 1));
        case UHADD: return u32((u64(a) + b) >> 1);
        case URHADD: return u32((u64(a) + b + 1) >> 1);
        case MINMAX_I: return u64(u32(std::min(sa, sb))) << 32 | u32(std::max(sa, int(c)));
        case MINMAX_U: return u64(std::min(a, b)) << 32 | std::max(b, c);
        case MINMAX_LL: return u64(std::min(i64(A), i64(B))) ^ (u64(std::max(i64(A), i64(B))) << 1);
        case ABS_I: return u32(sa == INT32_MIN ? 0 : std::abs(sa));
        default: return u64(i64(A) == INT64_MIN ? 1 : std::llabs(i64(A)));
    }
}

struct Data {
    std::vector<u32> a, b, c;
    std::vector<u64> A, B;
};

template <int F>
static void run(const Data& d, stress::Buffer<u32>& da, stress::Buffer<u32>& db, stress::Buffer<u32>& dc, stress::Buffer<u64>& dA,
                stress::Buffer<u64>& dB, stress::Buffer<u64>& dout) {
    using namespace stress;
    int n = int(d.a.size());
    std::vector<u64> want(n);
    double ref = best_ms([&] {
#pragma omp parallel for
        for (int i = 0; i < n; ++i) want[i] = ref_op(F, d.a[i], d.b[i], d.c[i], d.A[i], d.B[i]);
    });
    dout.zero();
    double k = time_device([&] { k_op<F><<<(n + 255) / 256, 256>>>(da, db, dc, dA, dB, dout, n); });
    auto got = dout.download();
    bool ok = true;
    for (int i = 0; i < n && ok; ++i)
        if (got[i] != want[i])
            ok = false, std::printf("  mismatch [%d] a=%08x b=%08x c=%08x A=%016llx B=%016llx: got %llx want %llx\n", i, d.a[i], d.b[i],
                                    d.c[i], d.A[i], d.B[i], got[i], want[i]);
    report(names[F], ok, k, ref);
}

template <int... F>
static void run_all(std::integer_sequence<int, F...>, const Data& d, stress::Buffer<u32>& da, stress::Buffer<u32>& db,
                    stress::Buffer<u32>& dc, stress::Buffer<u64>& dA, stress::Buffer<u64>& dB, stress::Buffer<u64>& dout) {
    (run<F>(d, da, db, dc, dA, dB, dout), ...);
}

int main() {
    const int n = 1 << 18;
    Data d;
    d.a = stress::random_u32(n, 1), d.b = stress::random_u32(n, 2), d.c = stress::random_u32(n, 3);
    auto h = stress::random_u32(n, 4), l = stress::random_u32(n, 5), r = stress::random_u32(n, 6);
    d.A.resize(n), d.B.resize(n);
    const u32 sp32[] = {0u, 1u, 0xffffffffu, 0x80000000u, 0x7fffffffu, 0x00800000u, 0x00ffffffu, 0xff000000u, 0x80808080u, 0x7f7f7f7fu};
    const u64 sp64[] = {0ull, 1ull, ~0ull, 1ull << 63, (1ull << 63) - 1, 0xffffffffull, 1ull << 32, 0x8000000080000000ull};
    for (int i = 0; i < n; ++i) {
        d.A[i] = u64(h[i]) << 32 | l[i];
        d.B[i] = u64(r[i]) << 32 | h[(i + 1) % n];
        // Sparse words exercise clz/ffs over the whole range.
        if (i % 4 == 1) d.a[i] >>= (r[i] % 32), d.A[i] >>= (r[i] % 64);
        if (i % 4 == 2) d.a[i] <<= (l[i] % 32), d.A[i] <<= (l[i] % 64);
        if (i % 13 == 0) d.a[i] = sp32[(i / 13) % 10];
        if (i % 17 == 0) d.b[i] = sp32[(i / 17) % 10];
        if (i % 11 == 0) d.A[i] = sp64[(i / 11) % 8];
        if (i % 19 == 0) d.B[i] = sp64[(i / 19) % 8];
    }
    stress::Buffer<u32> da(d.a), db(d.b), dc(d.c);
    stress::Buffer<u64> dA(d.A), dB(d.B), dout(n);
    run_all(std::make_integer_sequence<int, NFN>{}, d, da, db, dc, dA, dB, dout);
    return stress::finish();
}
