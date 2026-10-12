// Single precision special functions: erf, erfc, erfinv, erfcinv, erfcx,
// lgamma, tgamma, j0, j1, y0, y1, normcdf, normcdfinv, norm3d, rnorm3d,
// cyl_bessel_i0. Bounds from the CUDA Programming Guide (single precision).
#include "math_common.cuh"

using mc::ld;
#define R1(E) [](ld x, ld, ld) -> ld { return E; }
#define R3(E) [](ld x, ld y, ld z) -> ld { return E; }

static const ld SQRT2 = 1.414213562373095048801688724209698079L;
static const ld SQRTPI = 1.772453850905516027298167483341145183L;

// erf(y) = x for |x| <= 0.5 by Newton from the linear term (erf is accurate there).
static ld erfinv_small(ld x) {
    if (x == 0) return x;
    ld y = x * SQRTPI / 2;
    for (int k = 0; k < 8; ++k) y -= (std::erf(y) - x) / (2 / SQRTPI * std::exp(-y * y));
    return y;
}
// erfc(y) = t for t in (0, 2): bisection then Newton, in long double.
static ld erfc_solve(ld t) {
    if (std::isnan(t) || t < 0 || t > 2) return NAN;
    if (t == 0) return INFINITY;
    if (t == 2) return -INFINITY;
    if (t == 1) return 0;
    if (t > 1) return -erfc_solve(2 - t);  // exact for t of a narrower type
    if (t > 0.5L) return erfinv_small(1 - t);  // erfc(y) = t <=> erf(y) = 1 - t (exact), accurate for small y
    ld lo = 0, hi = 110;
    for (int k = 0; k < 40; ++k) {
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
    if (x == 0) return x;  // keeps the sign of zero
    if (x < 0) return -erfinv_l(-x);
    if (x > 0.5L) return erfc_solve(1 - x);  // exact: x is a float or double in (0.5, 1]
    return erfinv_small(x);
}
static ld erfcx_l(ld x) {
    if (x < 50) return std::exp(x * x) * std::erfc(x);
    // Asymptotic series for large x: 1/(x sqrt(pi)) * (1 - 1/(2x^2) + 3/(4x^4) - 15/(8x^6))
    ld u = 1 / (2 * x * x);
    return 1 / (x * SQRTPI) * (1 - u + 3 * u * u - 15 * u * u * u);
}
// normcdfinv(p) = -sqrt(2) erfcinv(2p).
static ld normcdfinv_l(ld p) {
    if (std::isnan(p) || p < 0 || p > 1) return NAN;
    return -SQRT2 * erfc_solve(2 * p) + 0;  // normcdfinv(0.5) = +0
}
// Modified Bessel I0 by its power series (converges for all x; terms stay finite for |x| <= 80).
static ld i0_l(ld x) {
    if (std::isnan(x)) return NAN;
    if (std::isinf(x)) return INFINITY;
    ld q = x * x / 4, term = 1, s = 1;
    for (int k = 1; k < 400; ++k) {
        term *= q / ((ld)k * k);
        s += term;
        if (term < s * 1e-22L) break;
    }
    return s;
}

MC_FN(Erf, float, erff(x));
MC_FN(Erfc, float, erfcf(x));
MC_FN(Erfinv, float, erfinvf(x));
MC_FN(Erfcinv, float, erfcinvf(x));
MC_FN(Erfcx, float, erfcxf(x));
MC_FN(Lgamma, float, lgammaf(x));
MC_FN(Tgamma, float, tgammaf(x));
MC_FN(J0, float, j0f(x));
MC_FN(J1, float, j1f(x));
MC_FN(Y0, float, y0f(x));
MC_FN(Y1, float, y1f(x));
MC_FN(Normcdf, float, normcdff(x));
MC_FN(Normcdfinv, float, normcdfinvf(x));
MC_FN(Norm3d, float, norm3df(x, y, z));
MC_FN(Rnorm3d, float, rnorm3df(x, y, z));
MC_FN(I0, float, cyl_bessel_i0f(x));

int main() {
    using namespace mc;
    const int n = 1 << 16;
    const float MX = FLT_MAX;
    run<float>("erff", Erf{}, R1(std::erf(x)), {2}, sweep<float>(n, -5, 5, 1, true, MX));
    run<float>("erfcf", Erfc{}, R1(std::erfc(x)), {4}, sweep<float>(n, -5, 11, 2, true, MX));
    run<float>("erfinvf", Erfinv{}, R1(erfinv_l(x)), {2}, sweep<float>(n, -1, 1, 3));
    run<float>("erfcinvf", Erfcinv{}, R1(erfc_solve(x)), {4}, sweep<float>(n, 0, 2, 4));
    run<float>("erfcxf", Erfcx{}, R1(erfcx_l(x)), {4}, sweep<float>(n, -9, 100, 5, true, 1e30f));
    {
        // lgammaf: 6 ulp outside (-10.001, -2.264); inputs there are moved out.
        auto x = sweep<float>(n, -30, 40, 6, true, MX);
        for (auto& v : x)
            if (v > -10.001f && v < -2.264f) v -= 10.0f;
        run<float>("lgammaf", Lgamma{}, R1(std::lgamma(x)), {6}, x);
    }
    run<float>("tgammaf", Tgamma{}, R1(std::tgamma(x)), {5}, sweep<float>(n, -30, 36, 7));
    {
        // Bessel functions: 9 ulp for |x| < 8, absolute 2.2e-6 beyond.
        auto bessel = [](ld x, ld, double& u, double& a) { u = 9, a = std::fabs(x) < 8 ? 0 : 2.2e-6; };
        auto xs = sweep<float>(n, -60, 60, 8, true, 1e30f), xp = sweep<float>(n, 0, 60, 9, true, 1e30f);
        run<float>("j0f", J0{}, R1(j0l(x)), {9, 0, bessel}, xs);
        run<float>("j1f", J1{}, R1(j1l(x)), {9, 0, bessel}, xs);
        run<float>("y0f", Y0{}, R1(y0l(x)), {9, 0, bessel}, xp);
        run<float>("y1f", Y1{}, R1(y1l(x)), {9, 0, bessel}, xp);
    }
    run<float>("normcdff", Normcdf{}, R1(std::erfc(-x / SQRT2) / 2), {5}, sweep<float>(n, -14, 6, 10, true, MX));
    run<float>("normcdfinvf", Normcdfinv{}, R1(normcdfinv_l(x)), {5}, sweep<float>(n, 0, 1, 11));
    {
        auto a = sweep<float>(n, -1e3f, 1e3f, 12, true, 1e30f), b = sweep<float>(n, -1e3f, 1e3f, 13, true, 1e30f),
             c = sweep<float>(n, -1e3f, 1e3f, 14, true, 1e30f);
        auto norm = [](ld x, ld y, ld z) -> ld {
            if (std::isinf(x) || std::isinf(y) || std::isinf(z)) return INFINITY;
            return std::sqrt(x * x + y * y + z * z);
        };
        auto rnorm = [](ld x, ld y, ld z) -> ld {
            if (std::isinf(x) || std::isinf(y) || std::isinf(z)) return 0;
            return 1 / std::sqrt(x * x + y * y + z * z);
        };
        run<float>("norm3df", Norm3d{}, norm, {3}, a, b, c);
        run<float>("rnorm3df", Rnorm3d{}, rnorm, {2}, a, b, c);
    }
    run<float>("cyl_bessel_i0f", I0{}, R1(i0_l(x)), {6}, sweep<float>(n, -80, 80, 15));
    return stress::finish();
}
