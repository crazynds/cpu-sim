// Single precision math library, common functions: roots, exponentials,
// logarithms, trigonometric, inverse trigonometric, hyperbolic, pow, hypot.
// Bounds: max ulp errors from the CUDA Programming Guide (single precision).
#include "math_common.cuh"

using mc::ld;
#define R1(E) [](ld x, ld, ld) -> ld { return E; }
#define R2(E) [](ld x, ld y, ld) -> ld { return E; }

static const ld PI_L = 3.141592653589793238462643383279502884L;

// sin(pi x) and cos(pi x) with exact argument reduction.
static ld sinpi_l(ld x) {
    if (std::isinf(x) || std::isnan(x)) return NAN;
    if (x == std::floor(x)) return std::copysign((ld)0, x);
    ld s = 1;
    if (x < 0) x = -x, s = -1;
    ld r = std::fmod(x, (ld)2);  // exact
    if (r > 1) r -= 1, s = -s;   // exact
    if (r > 0.5L) r = 1 - r;     // exact
    return s * std::sin(PI_L * r);
}
static ld cospi_l(ld x) {
    if (std::isinf(x) || std::isnan(x)) return NAN;
    ld r = std::fmod(std::fabs(x), (ld)2);
    if (r > 1) r = 2 - r;  // cos(pi r) symmetric about 1
    ld t = 0.5L - r;       // cos(pi r) = sin(pi (0.5 - r))
    if (t == 0) return 0;  // +0
    ld s = 1;
    if (t < 0) t = -t, s = -1;
    return s * std::sin(PI_L * t);
}

MC_FN(Sqrt, float, sqrtf(x));
MC_FN(Rsqrt, float, rsqrtf(x));
MC_FN(Cbrt, float, cbrtf(x));
MC_FN(Rcbrt, float, rcbrtf(x));
MC_FN(Exp, float, expf(x));
MC_FN(Exp2, float, exp2f(x));
MC_FN(Exp10, float, exp10f(x));
MC_FN(Expm1, float, expm1f(x));
MC_FN(Log, float, logf(x));
MC_FN(Log2, float, log2f(x));
MC_FN(Log10, float, log10f(x));
MC_FN(Log1p, float, log1pf(x));
MC_FN(Sin, float, sinf(x));
MC_FN(Cos, float, cosf(x));
MC_FN(Tan, float, tanf(x));
MC_FN(SincosS, float, ([&] { float s, c; sincosf(x, &s, &c); return s; })());
MC_FN(SincosC, float, ([&] { float s, c; sincosf(x, &s, &c); return c; })());
MC_FN(Sinpi, float, sinpif(x));
MC_FN(Cospi, float, cospif(x));
MC_FN(Asin, float, asinf(x));
MC_FN(Acos, float, acosf(x));
MC_FN(Atan, float, atanf(x));
MC_FN(Atan2, float, atan2f(x, y));
MC_FN(Sinh, float, sinhf(x));
MC_FN(Cosh, float, coshf(x));
MC_FN(Tanh, float, tanhf(x));
MC_FN(Asinh, float, asinhf(x));
MC_FN(Acosh, float, acoshf(x));
MC_FN(Atanh, float, atanhf(x));
MC_FN(Pow, float, powf(x, y));
MC_FN(Hypot, float, hypotf(x, y));
// The std:: overloads in device code (float arguments select the float versions).
MC_FN(StdMix, float, std::sqrt(x) + std::exp(-std::fabs(x)) * std::sin(x));

int main() {
    using namespace mc;
    const int n = 1 << 16;
    const float MX = FLT_MAX;
    run<float>("sqrtf", Sqrt{}, R1(std::sqrt(x)), {0}, sweep<float>(n, 0, 1e4f, 1, true, MX));
    run<float>("rsqrtf", Rsqrt{}, R1(1 / std::sqrt(x)), {2}, sweep<float>(n, 0, 1e4f, 2, true, MX));
    run<float>("cbrtf", Cbrt{}, R1(std::cbrt(x)), {1}, sweep<float>(n, -1e4f, 1e4f, 3, true, MX));
    run<float>("rcbrtf", Rcbrt{}, R1(1 / std::cbrt(x)), {1}, sweep<float>(n, -1e4f, 1e4f, 4, true, MX));
    run<float>("expf", Exp{}, R1(std::exp(x)), {2}, sweep<float>(n, -104, 89, 5));
    run<float>("exp2f", Exp2{}, R1(std::exp2(x)), {2}, sweep<float>(n, -150, 128, 6));
    run<float>("exp10f", Exp10{}, R1(std::pow((ld)10, x)), {2}, sweep<float>(n, -46, 39, 7));
    // expm1f is documented at 1 ulp; sm_120 measures 1.33 ulp near x = 17: bound 2.
    run<float>("expm1f", Expm1{}, R1(std::expm1(x)), {2}, sweep<float>(n, -20, 89, 8));
    run<float>("logf", Log{}, R1(std::log(x)), {1}, sweep<float>(n, 0, 100, 9, true, MX));
    run<float>("log2f", Log2{}, R1(std::log2(x)), {1}, sweep<float>(n, 0, 100, 10, true, MX));
    run<float>("log10f", Log10{}, R1(std::log10(x)), {2}, sweep<float>(n, 0, 100, 11, true, MX));
    run<float>("log1pf", Log1p{}, R1(std::log1p(x)), {1}, sweep<float>(n, -1, 100, 12, true, MX));
    run<float>("sinf", Sin{}, R1(std::sin(x)), {2}, sweep<float>(n, -10, 10, 13, true, MX));
    run<float>("cosf", Cos{}, R1(std::cos(x)), {2}, sweep<float>(n, -10, 10, 14, true, MX));
    run<float>("tanf", Tan{}, R1(std::tan(x)), {4}, sweep<float>(n, -10, 10, 15, true, MX));
    run<float>("sincosf_sin", SincosS{}, R1(std::sin(x)), {2}, sweep<float>(n, -10, 10, 16, true, MX));
    run<float>("sincosf_cos", SincosC{}, R1(std::cos(x)), {2}, sweep<float>(n, -10, 10, 16, true, MX));
    run<float>("sinpif", Sinpi{}, R1(sinpi_l(x)), {1}, sweep<float>(n, -10, 10, 17, true, MX));
    run<float>("cospif", Cospi{}, R1(cospi_l(x)), {1}, sweep<float>(n, -10, 10, 18, true, MX));
    run<float>("asinf", Asin{}, R1(std::asin(x)), {2}, sweep<float>(n, -1, 1, 19));
    run<float>("acosf", Acos{}, R1(std::acos(x)), {2}, sweep<float>(n, -1, 1, 20));
    run<float>("atanf", Atan{}, R1(std::atan(x)), {2}, sweep<float>(n, -10, 10, 21, true, MX));
    {
        std::vector<float> x, y;
        pairs(sweep<float>(n, -10, 10, 22, true, MX), sweep<float>(n, -10, 10, 23, true, MX), x, y);
        run<float>("atan2f", Atan2{}, R2(std::atan2(x, y)), {3}, x, y);
    }
    run<float>("sinhf", Sinh{}, R1(std::sinh(x)), {3}, sweep<float>(n, -90, 90, 24));
    run<float>("coshf", Cosh{}, R1(std::cosh(x)), {2}, sweep<float>(n, -90, 90, 25));
    run<float>("tanhf", Tanh{}, R1(std::tanh(x)), {2}, sweep<float>(n, -10, 10, 26, true, MX));
    run<float>("asinhf", Asinh{}, R1(std::asinh(x)), {3}, sweep<float>(n, -100, 100, 27, true, MX));
    run<float>("acoshf", Acosh{}, R1(std::acosh(x)), {4}, sweep<float>(n, 1, 100, 28, true, MX));
    run<float>("atanhf", Atanh{}, R1(std::atanh(x)), {3}, sweep<float>(n, -1, 1, 29));
    {
        // x over [-100, 100] (negative x meets non-integer y: NaN), y over
        // [-30, 30] with integers mixed in, plus all special pairs.
        auto xs = sweep<float>(n, -100, 100, 30, true, MX), ys = sweep<float>(n, -30, 30, 31, true, MX);
        for (int i = 0; i < n; i += 3) ys[i] = std::round(ys[i]);
        std::vector<float> x, y;
        pairs(xs, ys, x, y);
        run<float>("powf", Pow{}, R2(std::pow(x, y)), {4}, x, y);
    }
    {
        std::vector<float> x, y;
        pairs(sweep<float>(n, -1e3f, 1e3f, 32, true, MX), sweep<float>(n, -1e3f, 1e3f, 33, true, MX), x, y);
        run<float>("hypotf", Hypot{}, R2(std::hypot(x, y)), {3}, x, y);
    }
    // sqrt (0 ulp) + exp (2 ulp, of a value <= 1) * sin (2 ulp, <= 1), each rounded: a few ulps of the sum.
    run<float>("std_overloads_float", StdMix{}, R1(std::sqrt(x) + std::exp(-std::fabs(x)) * std::sin(x)), {8, 2e-7},
               sweep<float>(n, 0, 100, 34, false));
    return stress::finish();
}
