// Conversion and reinterpretation intrinsics, one case per function: bit
// casts, float/double to integer with each rounding mode, integer to float
// with each rounding mode, double to float with each rounding mode, float to
// half, __saturatef, rintf/nearbyintf/lrintf/llrintf/roundf/lroundf.
// Inputs stay in the destination's range; ties (k + 0.5) are frequent.
#include <cuda_fp16.h>

#include <cstring>

#include "stress.cuh"

using u64 = unsigned long long;
using i64 = long long;

enum Fn {
    F_AS_I, I_AS_F, D_AS_LL, LL_AS_D, F2I_RN, F2I_RZ, F2I_RU, F2I_RD, F2UI_RN, F2UI_RZ, F2LL_RN, F2LL_RD, D2I_RN, D2I_RZ,
    D2F_RN, D2F_RZ, D2F_RU, D2F_RD, I2F_RN, I2F_RZ, I2F_RU, I2F_RD, UI2F_RN, LL2F_RN, LL2F_RZ, LL2D_RN, I2D, F2H_RN, SAT,
    RINT, NEARBYINT, LRINT, LLRINT, ROUND, LROUND, NFN
};
static const char* names[NFN] = {
    "float_as_int", "int_as_float", "double_as_longlong", "longlong_as_double", "float2int_rn", "float2int_rz", "float2int_ru",
    "float2int_rd", "float2uint_rn", "float2uint_rz", "float2ll_rn", "float2ll_rd", "double2int_rn", "double2int_rz",
    "double2float_rn", "double2float_rz", "double2float_ru", "double2float_rd", "int2float_rn", "int2float_rz", "int2float_ru",
    "int2float_rd", "uint2float_rn", "ll2float_rn", "ll2float_rz", "ll2double_rn", "int2double_rn", "float2half_rn",
    "saturatef", "rintf", "nearbyintf", "lrintf", "llrintf", "roundf", "lroundf"};

struct In {
    float f;     // in int range
    float fu;    // in [0, 2^32)
    double d;    // any magnitude (double to float)
    double di;   // in int range (double to int)
    int i;
    i64 ll;
};

__device__ __host__ inline u64 fbits(float f) {
    u64 r = 0;
    std::memcpy(&r, &f, 4);
    return r;
}

template <int F>
__device__ u64 dev_op(In x) {
    if constexpr (F == F_AS_I) return unsigned(__float_as_int(x.f));
    else if constexpr (F == I_AS_F) return __float_as_uint(__int_as_float(x.i) * 1.0f);
    else if constexpr (F == D_AS_LL) return u64(__double_as_longlong(x.d));
    else if constexpr (F == LL_AS_D) return u64(__double_as_longlong(__longlong_as_double(x.ll & 0x7fefffffffffffffll) * 2.0));
    else if constexpr (F == F2I_RN) return unsigned(__float2int_rn(x.f));
    else if constexpr (F == F2I_RZ) return unsigned(__float2int_rz(x.f));
    else if constexpr (F == F2I_RU) return unsigned(__float2int_ru(x.f));
    else if constexpr (F == F2I_RD) return unsigned(__float2int_rd(x.f));
    else if constexpr (F == F2UI_RN) return __float2uint_rn(x.fu);
    else if constexpr (F == F2UI_RZ) return __float2uint_rz(x.fu);
    else if constexpr (F == F2LL_RN) return u64(__float2ll_rn(x.f * 1024.0f));
    else if constexpr (F == F2LL_RD) return u64(__float2ll_rd(x.f * 1024.0f));
    else if constexpr (F == D2I_RN) return unsigned(__double2int_rn(x.di / 4.0));
    else if constexpr (F == D2I_RZ) return unsigned(__double2int_rz(x.di / 4.0));
    else if constexpr (F == D2F_RN) return __float_as_uint(__double2float_rn(x.d));
    else if constexpr (F == D2F_RZ) return __float_as_uint(__double2float_rz(x.d));
    else if constexpr (F == D2F_RU) return __float_as_uint(__double2float_ru(x.d));
    else if constexpr (F == D2F_RD) return __float_as_uint(__double2float_rd(x.d));
    else if constexpr (F == I2F_RN) return __float_as_uint(__int2float_rn(x.i));
    else if constexpr (F == I2F_RZ) return __float_as_uint(__int2float_rz(x.i));
    else if constexpr (F == I2F_RU) return __float_as_uint(__int2float_ru(x.i));
    else if constexpr (F == I2F_RD) return __float_as_uint(__int2float_rd(x.i));
    else if constexpr (F == UI2F_RN) return __float_as_uint(__uint2float_rn(unsigned(x.i)));
    else if constexpr (F == LL2F_RN) return __float_as_uint(__ll2float_rn(x.ll));
    else if constexpr (F == LL2F_RZ) return __float_as_uint(__ll2float_rz(x.ll));
    else if constexpr (F == LL2D_RN) return u64(__double_as_longlong(__ll2double_rn(x.ll)));
    else if constexpr (F == I2D) return u64(__double_as_longlong(__int2double_rn(x.i)));
    else if constexpr (F == F2H_RN) return __half_as_ushort(__float2half_rn(x.f * 1e-5f));
    else if constexpr (F == SAT) return __float_as_uint(__saturatef(__fadd_rn(__fmul_rn(x.f, 1e-9f), 0.5f)));  // no FMA contraction
    else if constexpr (F == RINT) return __float_as_uint(rintf(x.f));
    else if constexpr (F == NEARBYINT) return __float_as_uint(nearbyintf(x.f));
    else if constexpr (F == LRINT) return u64(lrintf(x.f));
    else if constexpr (F == LLRINT) return u64(llrintf(x.f * 1024.0f));
    else if constexpr (F == ROUND) return __float_as_uint(roundf(x.f));
    else return u64(lroundf(x.f));
}

template <int F>
__global__ void k_op(const In* in, u64* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = dev_op<F>(in[i]);
}

// Round an exactly known value v (long double holds every int64 and double)
// to float in the given direction: 0 nearest, 1 toward zero, 2 up, 3 down.
static float to_float(long double v, int mode) {
    float f = float(v);  // nearest
    if (mode == 1 && std::fabs((long double)f) > std::fabs(v)) f = std::nextafter(f, 0.0f);
    if (mode == 2 && (long double)f < v) f = std::nextafter(f, INFINITY);
    if (mode == 3 && (long double)f > v) f = std::nextafter(f, -INFINITY);
    return f;
}
static u64 fb(float f) { return fbits(f); }
static u64 db(double d) {
    u64 r;
    std::memcpy(&r, &d, 8);
    return r;
}
static long long rne(long double v) { return std::llrint(v); }  // default rounding: nearest even

static u64 ref_op(int f, const In& x) {
    switch (f) {
        case F_AS_I: return fb(x.f);
        case I_AS_F: {
            float g;
            std::memcpy(&g, &x.i, 4);
            g = g * 1.0f;
            return fb(g);
        }
        case D_AS_LL: return db(x.d);
        case LL_AS_D: {
            double g;
            i64 b = x.ll & 0x7fefffffffffffffll;
            std::memcpy(&g, &b, 8);
            return db(g * 2.0);
        }
        case F2I_RN: return unsigned(int(rne(x.f)));
        case F2I_RZ: return unsigned(int(std::trunc(x.f)));
        case F2I_RU: return unsigned(int(std::ceil(x.f)));
        case F2I_RD: return unsigned(int(std::floor(x.f)));
        case F2UI_RN: return unsigned(rne(x.fu));
        case F2UI_RZ: return unsigned(std::trunc(x.fu));
        case F2LL_RN: return u64(rne(x.f * 1024.0f));
        case F2LL_RD: return u64(i64(std::floor(x.f * 1024.0f)));
        case D2I_RN: return unsigned(int(rne(x.di / 4.0)));
        case D2I_RZ: return unsigned(int(std::trunc(x.di / 4.0)));
        case D2F_RN: return fb(to_float(x.d, 0));
        case D2F_RZ: return fb(to_float(x.d, 1));
        case D2F_RU: return fb(to_float(x.d, 2));
        case D2F_RD: return fb(to_float(x.d, 3));
        case I2F_RN: return fb(to_float(x.i, 0));
        case I2F_RZ: return fb(to_float(x.i, 1));
        case I2F_RU: return fb(to_float(x.i, 2));
        case I2F_RD: return fb(to_float(x.i, 3));
        case UI2F_RN: return fb(to_float((long double)unsigned(x.i), 0));
        case LL2F_RN: return fb(to_float((long double)x.ll, 0));
        case LL2F_RZ: return fb(to_float((long double)x.ll, 1));
        case LL2D_RN: return db(double(x.ll));
        case I2D: return db(double(x.i));
        case F2H_RN: {  // the bits by copy: __half_as_ushort is device-only before CUDA 12.2
            const __half h = __double2half(double(x.f * 1e-5f));
            unsigned short bits;
            std::memcpy(&bits, &h, sizeof(bits));
            return bits;
        }
        case SAT: {
            volatile float p = x.f * 1e-9f;  // rounded on its own, as on the device: no FMA contraction
            float v = p + 0.5f;
            return fb(std::isnan(v) ? 0.0f : std::min(1.0f, std::max(0.0f, v)));
        }
        case RINT:
        case NEARBYINT: return fb(float(std::nearbyint(x.f)));
        case LRINT: return u64(std::lrint(x.f));
        case LLRINT: return u64(std::llrint(x.f * 1024.0f));
        case ROUND: return fb(std::round(x.f));
        default: return u64(std::lround(x.f));
    }
}

template <int F>
static void run(const std::vector<In>& in, stress::Buffer<In>& din, stress::Buffer<u64>& dout) {
    using namespace stress;
    int n = int(in.size());
    std::vector<u64> want(n);
    double ref = best_ms([&] {
#pragma omp parallel for
        for (int i = 0; i < n; ++i) want[i] = ref_op(F, in[i]);
    });
    dout.zero();
    double k = time_device([&] { k_op<F><<<(n + 255) / 256, 256>>>(din, dout, n); });
    auto got = dout.download();
    bool ok = true;
    for (int i = 0; i < n && ok; ++i)
        if (got[i] != want[i])
            ok = false, std::printf("  mismatch [%d] f=%.9g fu=%.9g d=%.17g i=%d ll=%lld: got %llx want %llx\n", i, in[i].f, in[i].fu,
                                    in[i].d, in[i].i, in[i].ll, got[i], want[i]);
    report(names[F], ok, k, ref);
}
template <int... F>
static void run_all(std::integer_sequence<int, F...>, const std::vector<In>& in, stress::Buffer<In>& din, stress::Buffer<u64>& dout) {
    (run<F>(in, din, dout), ...);
}

int main() {
    const int n = 1 << 18;
    auto r = stress::random_u32(4 * size_t(n), 1);
    auto f = stress::random_floats(n, 2, -1, 1);
    std::vector<In> in(n);
    for (int i = 0; i < n; ++i) {
        In& x = in[i];
        int e = int(r[i] % 31);  // magnitude up to 2^30
        x.f = std::ldexp(f[i], e);
        if (i % 5 == 0) x.f = float(int(r[n + i] % 20001) - 10000) + 0.5f;  // ties
        if (i % 7 == 0) x.f = float(int(r[n + i] % 9) - 4) * 0.25f;         // small, quarter steps
        if (i % 11 == 0) x.f = (i & 1) ? -0.0f : 0.0f;
        x.fu = std::ldexp(std::fabs(f[i]), int(r[2 * n + i] % 33)) * 0.999f;
        if (i % 6 == 0) x.fu = float(r[2 * n + i] % 100000) + 0.5f;
        x.d = double(f[i]) * std::ldexp(1.0, int(r[3 * n + i] % 60) - 30) + double(f[(i + 1) % n]) * 1e-12;
        x.di = double(f[i]) * std::ldexp(1.0, int(r[3 * n + i] % 33)) + double(f[(i + 1) % n]) * 1e-6;
        if (i % 9 == 0) x.di = double(int(r[n + i] % 4001) - 2000) * 2.0 + 2.0;  // exact ties after /4
        if (i % 13 == 0) x.d = std::ldexp(double(f[i]), -140);                  // float denormal range
        if (i % 17 == 0) x.d = double(f[i]) * 3.5e38;                           // near float max (may overflow)
        x.i = int(r[i]);
        if (i % 3 == 0) x.i >>= (r[n + i] % 31);
        x.ll = (i64(r[2 * n + i]) << 32 | r[3 * n + i]) >> (r[i] % 63);
    }
    stress::Buffer<In> din(in);
    stress::Buffer<u64> dout(n);
    run_all(std::make_integer_sequence<int, NFN>{}, in, din, dout);
    return stress::finish();
}
