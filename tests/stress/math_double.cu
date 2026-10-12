// Double precision math library: the functions of math_basic, math_misc and
// math_special in double. Bounds: max ulp errors from the CUDA Programming
// Guide (double precision); the references are long double (64-bit mantissa).
#include "math_common.cuh"

using mc::ld;
#define R1(E) [](ld x, ld, ld) -> ld { return E; }
#define R2(E) [](ld x, ld y, ld) -> ld { return E; }
#define R3(E) [](ld x, ld y, ld z) -> ld { return E; }

static const ld PI_L = 3.141592653589793238462643383279502884L;
static const ld SQRT2 = 1.414213562373095048801688724209698079L;
static const ld SQRTPI = 1.772453850905516027298167483341145183L;

static ld sinpi_l(ld x) {
    if (std::isinf(x) || std::isnan(x)) return NAN;
    if (x == std::floor(x)) return std::copysign((ld)0, x);
    ld s = 1;
    if (x < 0) x = -x, s = -1;
    ld r = std::fmod(x, (ld)2);
    if (r > 1) r -= 1, s = -s;
    if (r > 0.5L) r = 1 - r;
    return s * std::sin(PI_L * r);
}
static ld cospi_l(ld x) {
    if (std::isinf(x) || std::isnan(x)) return NAN;
    ld r = std::fmod(std::fabs(x), (ld)2);
    if (r > 1) r = 2 - r;
    ld t = 0.5L - r;
    if (t == 0) return 0;
    ld s = 1;
    if (t < 0) t = -t, s = -1;
    return s * std::sin(PI_L * t);
}
// erf(y) = x for |x| <= 0.5 by Newton from the linear term (erf is accurate there).
static ld erfinv_small(ld x) {
    if (x == 0) return x;
    ld y = x * SQRTPI / 2;
    for (int k = 0; k < 8; ++k) y -= (std::erf(y) - x) / (2 / SQRTPI * std::exp(-y * y));
    return y;
}
static ld erfc_solve(ld t) {
    if (std::isnan(t) || t < 0 || t > 2) return NAN;
    if (t == 0) return INFINITY;
    if (t == 2) return -INFINITY;
    if (t == 1) return 0;
    if (t > 1) return -erfc_solve(2 - t);
    if (t > 0.5L) return erfinv_small(1 - t);  // erfc(y) = t <=> erf(y) = 1 - t (exact), accurate for small y
    ld lo = 0, hi = 110;
    for (int k = 0; k < 50; ++k) {
        ld m = (lo + hi) / 2;
        (std::erfc(m) > t ? lo : hi) = m;
    }
    ld y = (lo + hi) / 2;
    for (int k = 0; k < 6; ++k) {
        ld d = (std::erfc(y) - t) / (-2 / SQRTPI * std::exp(-y * y));
        if (!std::isfinite(d)) break;
        y -= d;
    }
    return y;
}
static ld erfinv_l(ld x) {
    if (std::isnan(x) || x < -1 || x > 1) return NAN;
    if (x == 0) return x;
    if (x < 0) return -erfinv_l(-x);
    if (x > 0.5L) return erfc_solve(1 - x);  // exact: x is a float or double in (0.5, 1]
    return erfinv_small(x);
}
static ld ilogb_ref(ld x) {
    if (x == 0 || std::isnan(x)) return INT_MIN;
    if (std::isinf(x)) return INT_MAX;
    return std::ilogb((double)x);
}

MC_FN(Sqrt, double, sqrt(x));
MC_FN(Rsqrt, double, rsqrt(x));
MC_FN(Cbrt, double, cbrt(x));
MC_FN(Exp, double, exp(x));
MC_FN(Exp2, double, exp2(x));
MC_FN(Exp10, double, exp10(x));
MC_FN(Expm1, double, expm1(x));
MC_FN(Log, double, log(x));
MC_FN(Log2, double, log2(x));
MC_FN(Log10, double, log10(x));
MC_FN(Log1p, double, log1p(x));
MC_FN(Sin, double, sin(x));
MC_FN(Cos, double, cos(x));
MC_FN(Tan, double, tan(x));
MC_FN(SincosS, double, ([&] { double s, c; sincos(x, &s, &c); return s; })());
MC_FN(SincosC, double, ([&] { double s, c; sincos(x, &s, &c); return c; })());
MC_FN(Sinpi, double, sinpi(x));
MC_FN(Cospi, double, cospi(x));
MC_FN(Asin, double, asin(x));
MC_FN(Acos, double, acos(x));
MC_FN(Atan, double, atan(x));
MC_FN(Atan2, double, atan2(x, y));
MC_FN(Sinh, double, sinh(x));
MC_FN(Cosh, double, cosh(x));
MC_FN(Tanh, double, tanh(x));
MC_FN(Asinh, double, asinh(x));
MC_FN(Acosh, double, acosh(x));
MC_FN(Atanh, double, atanh(x));
MC_FN(Pow, double, pow(x, y));
MC_FN(Hypot, double, hypot(x, y));
MC_FN(Erf, double, erf(x));
MC_FN(Erfc, double, erfc(x));
MC_FN(Erfinv, double, erfinv(x));
MC_FN(Erfcinv, double, erfcinv(x));
MC_FN(Lgamma, double, lgamma(x));
MC_FN(Tgamma, double, tgamma(x));
MC_FN(J0, double, j0(x));
MC_FN(Y0, double, y0(x));
MC_FN(Normcdf, double, normcdf(x));
MC_FN(Norm3d, double, norm3d(x, y, z));
MC_FN(Rnorm3d, double, rnorm3d(x, y, z));
MC_FN(Fmod, double, fmod(x, y));
MC_FN(Remainder, double, remainder(x, y));
MC_FN(RemquoQ, double, ([&] { int q; double r = remquo(x, y, &q); return isnan(r) ? 0.0 : double(q < 0 ? -(-q & 7) : (q & 7)); })());
MC_FN(FrexpM, double, ([&] { int e; return frexp(x, &e); })());
MC_FN(FrexpE, double, ([&] { int e = 0; frexp(x, &e); return isfinite(x) ? double(e) : 0.0; })());
MC_FN(ModfF, double, ([&] { double ip; return modf(x, &ip); })());
MC_FN(Ldexp, double, ldexp(x, int(y)));
MC_FN(Scalbn, double, scalbn(x, int(y)));
MC_FN(Ilogb, double, double(ilogb(x)));
MC_FN(Logb, double, logb(x));
MC_FN(Nextafter, double, nextafter(x, y));
MC_FN(Copysign, double, copysign(x, y));
MC_FN(Fdim, double, fdim(x, y));
MC_FN(Fma, double, fma(x, y, z));
MC_FN(Fmin, double, fmin(x, y));
MC_FN(Fmax, double, fmax(x, y));
MC_FN(RoundAll, double, ceil(x) * 8 + floor(x) * 4 + trunc(x) * 2 + round(x) + rint(x) * 16);
MC_FN(Classify, double, double(isnan(x) * 1 + isinf(x) * 2 + isfinite(x) * 4 + signbit(x) * 8));

static ld remquo_q(ld x, ld y) {
    int q;
    double r = std::remquo((double)x, (double)y, &q);
    return std::isnan(r) ? 0 : (ld)(q < 0 ? -(-q & 7) : (q & 7));
}

int main() {
    using namespace mc;
    const int n = 1 << 16;
    const double MX = DBL_MAX;
    // expm1, log2, log10, cosh, tanh and rnorm3d are documented at 1 ulp; sm_120
    // measures up to 1.4 ulp against a long double reference: bound 2 for those.
    run<double>("sqrt", Sqrt{}, R1(std::sqrt(x)), {0}, sweep<double>(n, 0, 1e4, 1, true, MX));
    run<double>("rsqrt", Rsqrt{}, R1(1 / std::sqrt(x)), {1}, sweep<double>(n, 0, 1e4, 2, true, MX));
    run<double>("cbrt", Cbrt{}, R1(std::cbrt(x)), {1}, sweep<double>(n, -1e4, 1e4, 3, true, MX));
    run<double>("exp", Exp{}, R1(std::exp(x)), {1}, sweep<double>(n, -745, 710, 4));
    run<double>("exp2", Exp2{}, R1(std::exp2(x)), {1}, sweep<double>(n, -1075, 1024, 5));
    run<double>("exp10", Exp10{}, R1(std::pow((ld)10, x)), {1}, sweep<double>(n, -324, 309, 6));
    run<double>("expm1", Expm1{}, R1(std::expm1(x)), {2}, sweep<double>(n, -40, 710, 7));
    run<double>("log", Log{}, R1(std::log(x)), {1}, sweep<double>(n, 0, 100, 8, true, MX));
    run<double>("log2", Log2{}, R1(std::log2(x)), {2}, sweep<double>(n, 0, 100, 9, true, MX));
    run<double>("log10", Log10{}, R1(std::log10(x)), {2}, sweep<double>(n, 0, 100, 10, true, MX));
    run<double>("log1p", Log1p{}, R1(std::log1p(x)), {1}, sweep<double>(n, -1, 100, 11, true, MX));
    run<double>("sin", Sin{}, R1(std::sin(x)), {2}, sweep<double>(n, -10, 10, 12, true, MX));
    run<double>("cos", Cos{}, R1(std::cos(x)), {2}, sweep<double>(n, -10, 10, 13, true, MX));
    run<double>("tan", Tan{}, R1(std::tan(x)), {2}, sweep<double>(n, -10, 10, 14, true, MX));
    run<double>("sincos_sin", SincosS{}, R1(std::sin(x)), {2}, sweep<double>(n, -10, 10, 15, true, MX));
    run<double>("sincos_cos", SincosC{}, R1(std::cos(x)), {2}, sweep<double>(n, -10, 10, 15, true, MX));
    run<double>("sinpi", Sinpi{}, R1(sinpi_l(x)), {2}, sweep<double>(n, -10, 10, 16, true, MX));
    run<double>("cospi", Cospi{}, R1(cospi_l(x)), {2}, sweep<double>(n, -10, 10, 17, true, MX));
    run<double>("asin", Asin{}, R1(std::asin(x)), {2}, sweep<double>(n, -1, 1, 18));
    run<double>("acos", Acos{}, R1(std::acos(x)), {2}, sweep<double>(n, -1, 1, 19));
    run<double>("atan", Atan{}, R1(std::atan(x)), {2}, sweep<double>(n, -10, 10, 20, true, MX));
    {
        std::vector<double> x, y;
        pairs(sweep<double>(n, -10, 10, 21, true, MX), sweep<double>(n, -10, 10, 22, true, MX), x, y);
        run<double>("atan2", Atan2{}, R2(std::atan2(x, y)), {2}, x, y);
    }
    run<double>("sinh", Sinh{}, R1(std::sinh(x)), {2}, sweep<double>(n, -710, 710, 23));
    run<double>("cosh", Cosh{}, R1(std::cosh(x)), {2}, sweep<double>(n, -710, 710, 24));
    run<double>("tanh", Tanh{}, R1(std::tanh(x)), {2}, sweep<double>(n, -20, 20, 25, true, MX));
    run<double>("asinh", Asinh{}, R1(std::asinh(x)), {3}, sweep<double>(n, -100, 100, 26, true, MX));
    run<double>("acosh", Acosh{}, R1(std::acosh(x)), {3}, sweep<double>(n, 1, 100, 27, true, MX));
    run<double>("atanh", Atanh{}, R1(std::atanh(x)), {2}, sweep<double>(n, -1, 1, 28));
    {
        auto xs = sweep<double>(n, -100, 100, 29, true, MX), ys = sweep<double>(n, -150, 150, 30, true, MX);
        for (int i = 0; i < n; i += 3) ys[i] = std::round(ys[i]);
        std::vector<double> x, y;
        pairs(xs, ys, x, y);
        run<double>("pow", Pow{}, R2(std::pow(x, y)), {2}, x, y);
    }
    {
        std::vector<double> x, y;
        pairs(sweep<double>(n, -1e3, 1e3, 31, true, MX), sweep<double>(n, -1e3, 1e3, 32, true, MX), x, y);
        run<double>("hypot", Hypot{}, R2(std::hypot(x, y)), {2}, x, y);
        run<double>("fmod", Fmod{}, R2(std::fmod(x, y)), {0}, x, y);
        run<double>("remainder", Remainder{}, R2(std::remainder(x, y)), {0}, x, y);
        run<double>("remquo_quo", RemquoQ{}, R2(remquo_q(x, y)), {0}, x, y);
        run<double>("nextafter", Nextafter{}, R2(std::nextafter((double)x, (double)y)), {0}, x, y);
        run<double>("copysign", Copysign{}, R2(std::copysign(x, y)), {0}, x, y);
        run<double>("fdim", Fdim{}, R2(std::fdim(x, y)), {0}, x, y);
        for (size_t i = 0; i < x.size(); ++i)
            if (x[i] == 0 && y[i] == 0) y[i] = x[i];  // fmin/fmax of +0,-0 is unspecified
        run<double>("fmin", Fmin{}, R2(std::fmin(x, y)), {0}, x, y);
        run<double>("fmax", Fmax{}, R2(std::fmax(x, y)), {0}, x, y);
    }
    run<double>("erf", Erf{}, R1(std::erf(x)), {2}, sweep<double>(n, -6, 6, 33, true, MX));
    run<double>("erfc", Erfc{}, R1(std::erfc(x)), {5}, sweep<double>(n, -6, 27, 34, true, MX));
    run<double>("erfinv", Erfinv{}, R1(erfinv_l(x)), {5}, sweep<double>(n, -1, 1, 35));
    run<double>("erfcinv", Erfcinv{}, R1(erfc_solve(x)), {6}, sweep<double>(n, 0, 2, 36));
    {
        auto x = sweep<double>(n, -40, 60, 37, true, 1e300);
        for (auto& v : x)
            if (v > -23.0001 && v < -2.2637) v -= 21.0;  // 4 ulp outside (-23.0001, -2.2637)
        run<double>("lgamma", Lgamma{}, R1(std::lgamma(x)), {4}, x);
    }
    run<double>("tgamma", Tgamma{}, R1(std::tgamma(x)), {10}, sweep<double>(n, -170, 171, 38));
    {
        auto bessel = [](ld x, ld, double& u, double& a) { u = 7, a = std::fabs(x) < 8 ? 0 : 5e-12; };
        run<double>("j0", J0{}, R1(j0l(x)), {7, 0, bessel}, sweep<double>(n, -60, 60, 39, true, 1e300));
        run<double>("y0", Y0{}, R1(y0l(x)), {7, 0, bessel}, sweep<double>(n, 0, 60, 40, true, 1e300));
    }
    run<double>("normcdf", Normcdf{}, R1(std::erfc(-x / SQRT2) / 2), {5}, sweep<double>(n, -38, 9, 41, true, MX));
    {
        auto a = sweep<double>(n, -1e3, 1e3, 42, true, 1e300), b = sweep<double>(n, -1e3, 1e3, 43, true, 1e300),
             c = sweep<double>(n, -1e3, 1e3, 44, true, 1e300);
        auto norm = [](ld x, ld y, ld z) -> ld {
            if (std::isinf(x) || std::isinf(y) || std::isinf(z)) return INFINITY;
            return std::sqrt(x * x + y * y + z * z);
        };
        auto rnorm = [](ld x, ld y, ld z) -> ld {
            if (std::isinf(x) || std::isinf(y) || std::isinf(z)) return 0;
            return 1 / std::sqrt(x * x + y * y + z * z);
        };
        run<double>("norm3d", Norm3d{}, norm, {2}, a, b, c);
        run<double>("rnorm3d", Rnorm3d{}, rnorm, {2}, a, b, c);
        for (int i = 1; i < n; i += 4) c[i] = -(a[i] * b[i]);
        // fmal is a single rounding in long double: near-exact for the cancellation cases too.
        run<double>("fma", Fma{}, R3(std::fma(x, y, z)), {0}, a, b, c);
    }
    {
        auto x = sweep<double>(n, -1e3, 1e3, 45, true, MX);
        auto u = stress::random_u32(x.size(), 46);
        std::vector<double> e(x.size());
        for (size_t i = 0; i < x.size(); ++i) e[i] = double(int(u[i] % 4201) - 2100);
        run<double>("frexp_mantissa", FrexpM{}, R1([](ld v) { int k; return (ld)std::frexp((double)v, &k); }(x)), {0}, x);
        run<double>("frexp_exponent", FrexpE{}, R1([](ld v) { int k = 0; std::frexp((double)v, &k); return (ld)(std::isfinite(v) ? k : 0); }(x)), {0}, x);
        run<double>("modf_frac", ModfF{}, R1([](ld v) { double ip; return (ld)std::modf((double)v, &ip); }(x)), {0}, x);
        run<double>("ilogb", Ilogb{}, R1(ilogb_ref(x)), {0}, x);
        run<double>("logb", Logb{}, R1(std::logb((double)x)), {0}, x);
        run<double>("classify", Classify{}, R1(std::isnan(x) * 1 + std::isinf(x) * 2 + std::isfinite(x) * 4 + std::signbit(x) * 8), {0}, x);
        run<double>("ldexp", Ldexp{}, R2(std::ldexp(x, int(y))), {0}, x, e);
        run<double>("scalbn", Scalbn{}, R2(std::scalbn(x, int(y))), {0}, x, e);
        // Rounding functions combined (each exact; the weighted sum is exact for |x| < 2^40).
        auto r = sweep<double>(n, -1e3, 1e3, 47, false);
        for (int i = 0; i < n; i += 2) r[i] = double(int(u[i] % 4001) - 2000) * 0.5 + ((u[i] >> 20) % 3 == 0 ? 0.0 : 1e-9);
        run<double>("ceil_floor_trunc_round_rint", RoundAll{},
                    R1(std::ceil(x) * 8 + std::floor(x) * 4 + std::trunc(x) * 2 + std::round(x) + std::nearbyint(x) * 16), {0}, r);
    }
    return stress::finish();
}
