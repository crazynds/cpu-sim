// Fast single precision intrinsics (__expf, __exp10f, __logf, __log2f,
// __log10f, __sinf, __cosf, __sincosf, __tanf, __powf, __fdividef) over the
// input ranges where NVIDIA documents their error, with those bounds; and the
// IEEE arithmetic intrinsics with an explicit rounding mode (__fadd_rz,
// __fmul_ru, __fdiv_rd, __fsqrt_ru, __frcp_rz, __fmaf_rd, __frsqrt_rn,
// __dadd_ru, __dmul_rd, __ddiv_rz, __dsqrt_rd...), checked exactly.
#include "math_common.cuh"

using mc::ld;
#define R1(E) [](ld x, ld, ld) -> ld { return E; }
#define R2(E) [](ld x, ld y, ld) -> ld { return E; }
#define R3(E) [](ld x, ld y, ld z) -> ld { return E; }

MC_FN(Expf, float, __expf(x));
MC_FN(Exp10f, float, __exp10f(x));
MC_FN(Logf, float, __logf(x));
MC_FN(Log2f, float, __log2f(x));
MC_FN(Log10f, float, __log10f(x));
MC_FN(Sinf, float, __sinf(x));
MC_FN(Cosf, float, __cosf(x));
MC_FN(SincosS, float, ([&] { float s, c; __sincosf(x, &s, &c); return s; })());
MC_FN(SincosC, float, ([&] { float s, c; __sincosf(x, &s, &c); return c; })());
MC_FN(Tanf, float, __tanf(x));
MC_FN(Powf, float, __powf(x, y));
MC_FN(Fdividef, float, __fdividef(x, y));

// Rounding-mode intrinsics.
MC_FN(FaddRz, float, __fadd_rz(x, y));
MC_FN(FaddRu, float, __fadd_ru(x, y));
MC_FN(FaddRd, float, __fadd_rd(x, y));
MC_FN(FsubRn, float, __fsub_rn(x, y));
MC_FN(FmulRz, float, __fmul_rz(x, y));
MC_FN(FmulRu, float, __fmul_ru(x, y));
MC_FN(FmulRd, float, __fmul_rd(x, y));
MC_FN(FdivRn, float, __fdiv_rn(x, y));
MC_FN(FdivRz, float, __fdiv_rz(x, y));
MC_FN(FdivRu, float, __fdiv_ru(x, y));
MC_FN(FdivRd, float, __fdiv_rd(x, y));
MC_FN(FsqrtRn, float, __fsqrt_rn(x));
MC_FN(FsqrtRz, float, __fsqrt_rz(x));
MC_FN(FsqrtRu, float, __fsqrt_ru(x));
MC_FN(FrcpRn, float, __frcp_rn(x));
MC_FN(FrcpRz, float, __frcp_rz(x));
MC_FN(FrsqrtRn, float, __frsqrt_rn(x));
MC_FN(FmafRz, float, __fmaf_rz(x, y, z));
MC_FN(FmafRu, float, __fmaf_ru(x, y, z));
MC_FN(FmafRd, float, __fmaf_rd(x, y, z));
MC_FN(DaddRz, double, __dadd_rz(x, y));
MC_FN(DaddRu, double, __dadd_ru(x, y));
MC_FN(DmulRd, double, __dmul_rd(x, y));
MC_FN(DmulRu, double, __dmul_ru(x, y));
MC_FN(DdivRz, double, __ddiv_rz(x, y));
MC_FN(DdivRd, double, __ddiv_rd(x, y));
MC_FN(DsqrtRu, double, __dsqrt_ru(x));
MC_FN(DsqrtRd, double, __dsqrt_rd(x));
MC_FN(DrcpRu, double, __drcp_ru(x));

// Round an approximation v of the exact value to T in a direction: 0 nearest,
// 1 toward zero, 2 up, 3 down. `sign` is the sign of (exact - v) when v is
// not exact (0 when v is exact), so the decision never relies on v alone.
template <class T>
static T directed(ld v, int sign, int mode) {
    T f = (T)v;
    if (std::isnan(v)) return f;
    // exact - f has the sign of (v - f) when v != f, else of `sign`.
    int s = (ld)f < v ? 1 : (ld)f > v ? -1 : sign;
    const T inf = std::numeric_limits<T>::infinity();
    if (mode == 1 && ((f > 0 && s < 0) || (f < 0 && s > 0))) f = std::nextafter(f, T(0));
    if (mode == 1 && std::isinf(f) && !std::isinf(v)) f = std::nextafter(f, T(0));
    if (mode == 2 && s > 0) f = std::nextafter(f, inf);
    if (mode == 3 && s < 0) f = std::nextafter(f, -inf);
    return f;
}
// Signed zero of an exact zero sum: the common sign of two zeros, else -0 only when rounding down.
static ld zero_sum(ld x, ld y, int mode) {
    if (x == 0 && y == 0 && std::signbit(x) == std::signbit(y)) return x;  // (+0)+(+0), (-0)+(-0)
    return mode == 3 ? -0.0L : 0.0L;
}
// float: long double (64-bit mantissa) holds x*y of floats exactly, sums get
// the direction of their rounding from a two-sum; the quotient and root are
// never within 2^-64 of a float without being equal to it.
// Two-sum in long double: s + err == x + y exactly (finite operands).
static int sum_dir(ld x, ld y, ld s) {
    if (!std::isfinite(s)) return 0;
    ld bb = s - x, err = (x - (s - bb)) + (y - bb);
    return err > 0 ? 1 : err < 0 ? -1 : 0;
}
template <int Mode>
static ld f_add(ld x, ld y, ld) {
    ld s = x + y;
    if (s == 0 && !std::isnan(s)) return zero_sum(x, y, Mode);
    return directed<float>(s, sum_dir(x, y, s), Mode);
}
template <int Mode>
static ld f_mul(ld x, ld y, ld) { return directed<float>(x * y, 0, Mode); }
template <int Mode>
static ld f_div(ld x, ld y, ld) { return directed<float>(x / y, 0, Mode); }
template <int Mode>
static ld f_sqrt(ld x, ld, ld) { return directed<float>(std::sqrt(x), 0, Mode); }
template <int Mode>
static ld f_rcp(ld x, ld, ld) { return directed<float>(1 / x, 0, Mode); }
template <int Mode>
static ld f_fma(ld x, ld y, ld z) {
    ld p = x * y, s = p + z;  // p exact (48 bits); the two-sum gives the direction of the rounding of s
    if (s == 0 && !std::isnan(s)) return zero_sum(p, z, Mode);
    return directed<float>(s, sum_dir(p, z, s), Mode);
}
// double: exact error terms by fma give the direction.
template <int Mode>
static ld d_add(ld lx, ld ly, ld) {
    double x = (double)lx, y = (double)ly, s = x + y;
    if (!std::isfinite(s)) return directed<double>((ld)x + (ld)y, 0, Mode);
    if (s == 0) return zero_sum(x, y, Mode);
    double bb = s - x, err = (x - (s - bb)) + (y - bb);  // two-sum: exact = s + err
    return directed<double>(s, err > 0 ? 1 : err < 0 ? -1 : 0, Mode);
}
template <int Mode>
static ld d_mul(ld lx, ld ly, ld) {
    double x = (double)lx, y = (double)ly, p = x * y;
    if (!std::isfinite(p) || p == 0) return directed<double>((ld)x * (ld)y, 0, Mode);
    double err = std::fma(x, y, -p);
    return directed<double>(p, err > 0 ? 1 : err < 0 ? -1 : 0, Mode);
}
template <int Mode>
static ld d_div(ld lx, ld ly, ld) {
    double x = (double)lx, y = (double)ly, q = x / y;
    if (!std::isfinite(q) || q == 0) return directed<double>((ld)x / (ld)y, 0, Mode);
    double r = std::fma(-q, y, x);  // exact remainder: x - q y
    int s = (r > 0) == (y > 0) ? 1 : -1;
    return directed<double>(q, r == 0 ? 0 : s, Mode);
}
template <int Mode>
static ld d_sqrt(ld lx, ld, ld) {
    double x = (double)lx, q = std::sqrt(x);
    if (!std::isfinite(q) || q == 0) return directed<double>(std::sqrt(lx), 0, Mode);
    double r = std::fma(-q, q, x);
    return directed<double>(q, r > 0 ? 1 : r < 0 ? -1 : 0, Mode);
}

// Documented error bounds of the fast intrinsics. The ulp bounds of the
// logarithms outside [0.5, 2] get +0.5: sm_120 measures 3.02 ulp for __logf
// (documented 3) against the exact value, i.e. 3 ulp from the rounded one.
static void b_exp(ld x, ld, double& u, double& a) { u = 2 + std::floor(std::fabs(1.173 * (double)x)), a = 0; }
static void b_exp10(ld x, ld, double& u, double& a) { u = 2 + std::floor(std::fabs(2.97 * (double)x)), a = 0; }
static void b_log(ld x, ld, double& u, double& a) {
    bool in = x >= 0.5L && x <= 2;
    u = in ? 0 : 3.5, a = in ? std::exp2(-21.41) : 0;
}
static void b_log2(ld x, ld, double& u, double& a) {
    bool in = x >= 0.5L && x <= 2;
    u = in ? 0 : 2.5, a = in ? std::exp2(-22.0) : 0;
}
static void b_log10(ld x, ld, double& u, double& a) {
    bool in = x >= 0.5L && x <= 2;
    u = in ? 0 : 3.5, a = in ? std::exp2(-24.0) : 0;
}
static void b_sin(ld, ld, double& u, double& a) { u = 0, a = std::exp2(-21.41); }
static void b_cos(ld, ld, double& u, double& a) { u = 0, a = std::exp2(-21.19); }
// __tanf = __sinf * (1 / __cosf) on [-pi/4, pi/4] (cos >= 0.7): propagated sin/cos errors plus the rounding.
static void b_tan(ld, ld, double& u, double& a) { u = 2, a = (std::exp2(-21.41) + std::exp2(-21.19)) / 0.7 * 1.5; }
// __powf = exp2(y * __log2f(x)): error of the exponent t times ln 2, relative, plus 2 ulp of exp2.
static void b_pow(ld x, ld y, double& u, double& a) {
    double l = std::fabs(std::log2((double)x)), t = std::fabs((double)y) * l;
    double dt = std::fabs((double)y) * std::exp2(-22.0) * (1 + l) + t * std::exp2(-24.0);
    u = 3 + 0.6931 * dt * std::exp2(24.0), a = 0;
}

int main() {
    using namespace mc;
    const int n = 1 << 16;
    const float PI = 3.14159265f;
    run<float>("fast_expf", Expf{}, R1(std::exp(x)), {0, 0, b_exp}, sweep<float>(n, -87, 88, 1, false));
    run<float>("fast_exp10f", Exp10f{}, R1(std::pow((ld)10, x)), {0, 0, b_exp10}, sweep<float>(n, -37, 38, 2, false));
    auto pos = sweep<float>(n, 0.25f, 4.0f, 3, false, 1e38f);
    for (auto& v : pos) v = std::max(v, FLT_MIN);  // normal inputs only (the approximations flush denormals)
    run<float>("fast_logf", Logf{}, R1(std::log(x)), {0, 0, b_log}, pos);
    run<float>("fast_log2f", Log2f{}, R1(std::log2(x)), {0, 0, b_log2}, pos);
    run<float>("fast_log10f", Log10f{}, R1(std::log10(x)), {0, 0, b_log10}, pos);
    auto ang = sweep<float>(n, -PI, PI, 4, false);
    run<float>("fast_sinf", Sinf{}, R1(std::sin(x)), {0, 0, b_sin}, ang);
    run<float>("fast_cosf", Cosf{}, R1(std::cos(x)), {0, 0, b_cos}, ang);
    run<float>("fast_sincosf_sin", SincosS{}, R1(std::sin(x)), {0, 0, b_sin}, ang);
    run<float>("fast_sincosf_cos", SincosC{}, R1(std::cos(x)), {0, 0, b_cos}, ang);
    run<float>("fast_tanf", Tanf{}, R1(std::tan(x)), {0, 0, b_tan}, sweep<float>(n, -PI / 4, PI / 4, 5, false));
    {
        auto x = sweep<float>(n, 0.5f, 4.0f, 6, false), y = sweep<float>(n, -8, 8, 7, false);
        run<float>("fast_powf", Powf{}, R2(std::pow(x, y)), {0, 0, b_pow}, x, y);
    }
    {
        auto x = sweep<float>(n, -1e3f, 1e3f, 8, false, 1e18f), y = sweep<float>(n, -1e3f, 1e3f, 9, false, 1e18f);
        for (auto& v : x)
            if (std::fabs(v) < 1e-18f) v = 1e-18f;
        for (auto& v : y)
            if (std::fabs(v) < 1e-18f) v = -1e-18f;
        run<float>("fast_fdividef", Fdividef{}, R2(x / y), {2}, x, y);
    }

    // Rounding-mode arithmetic, float, including denormals and special values.
    {
        auto a = sweep<float>(n, -100, 100, 10), b = sweep<float>(n, -100, 100, 11), c = sweep<float>(n, -1e4f, 1e4f, 12);
        std::vector<float> x, y;
        pairs(a, b, x, y);
        run<float>("fadd_rz", FaddRz{}, f_add<1>, {0}, x, y);
        run<float>("fadd_ru", FaddRu{}, f_add<2>, {0}, x, y);
        run<float>("fadd_rd", FaddRd{}, f_add<3>, {0}, x, y);
        run<float>("fsub_rn", FsubRn{}, R2(f_add<0>(x, -y, 0)), {0}, x, y);
        run<float>("fmul_rz", FmulRz{}, f_mul<1>, {0}, x, y);
        run<float>("fmul_ru", FmulRu{}, f_mul<2>, {0}, x, y);
        run<float>("fmul_rd", FmulRd{}, f_mul<3>, {0}, x, y);
        run<float>("fdiv_rn", FdivRn{}, f_div<0>, {0}, x, y);
        run<float>("fdiv_rz", FdivRz{}, f_div<1>, {0}, x, y);
        run<float>("fdiv_ru", FdivRu{}, f_div<2>, {0}, x, y);
        run<float>("fdiv_rd", FdivRd{}, f_div<3>, {0}, x, y);
        auto p = sweep<float>(n, 0, 1e4f, 13, true, FLT_MAX);
        run<float>("fsqrt_rn", FsqrtRn{}, f_sqrt<0>, {0}, p);
        run<float>("fsqrt_rz", FsqrtRz{}, f_sqrt<1>, {0}, p);
        run<float>("fsqrt_ru", FsqrtRu{}, f_sqrt<2>, {0}, p);
        run<float>("frcp_rn", FrcpRn{}, f_rcp<0>, {0}, a);
        run<float>("frcp_rz", FrcpRz{}, f_rcp<1>, {0}, a);
        run<float>("frsqrt_rn", FrsqrtRn{}, R1(directed<float>(1 / std::sqrt(x), 0, 0)), {0}, p);
        size_t m = std::min({a.size(), b.size(), c.size()}) - 30;  // finite inputs only for the fma
        std::vector<float> fa(a.begin(), a.begin() + m), fb(b.begin(), b.begin() + m), fc(c.begin(), c.begin() + m);
        run<float>("fmaf_rz", FmafRz{}, f_fma<1>, {0}, fa, fb, fc);
        run<float>("fmaf_ru", FmafRu{}, f_fma<2>, {0}, fa, fb, fc);
        run<float>("fmaf_rd", FmafRd{}, f_fma<3>, {0}, fa, fb, fc);
    }
    // Double, finite normal values in a moderate range (the exact error terms
    // computed with fma need no underflow).
    {
        auto a = sweep<double>(n, -1e3, 1e3, 14, false, 1e100), b = sweep<double>(n, -1e3, 1e3, 15, false, 1e100);
        for (auto* v : {&a, &b})
            for (auto& e : *v)
                if (std::fabs(e) < 1e-100) e = std::copysign(1e-100, e == 0 ? 1.0 : e);
        run<double>("dadd_rz", DaddRz{}, d_add<1>, {0}, a, b);
        run<double>("dadd_ru", DaddRu{}, d_add<2>, {0}, a, b);
        run<double>("dmul_rd", DmulRd{}, d_mul<3>, {0}, a, b);
        run<double>("dmul_ru", DmulRu{}, d_mul<2>, {0}, a, b);
        run<double>("ddiv_rz", DdivRz{}, d_div<1>, {0}, a, b);
        run<double>("ddiv_rd", DdivRd{}, d_div<3>, {0}, a, b);
        for (auto& e : a) e = std::fabs(e);
        run<double>("dsqrt_ru", DsqrtRu{}, d_sqrt<2>, {0}, a);
        run<double>("dsqrt_rd", DsqrtRd{}, d_sqrt<3>, {0}, a);
        run<double>("drcp_ru", DrcpRu{}, R1(d_div<2>(1, x, 0)), {0}, b);
    }
    return stress::finish();
}
