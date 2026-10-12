// Single precision exact functions (0 ulp: correctly rounded or exact by
// definition): fmod, remainder, remquo, frexp, ldexp, modf, scalbn, ilogb,
// logb, nextafter, copysign, fdim, fma, fmin, fmax, ceil, floor, trunc,
// round, rint, isnan/isinf/isfinite/signbit.
#include "math_common.cuh"

using mc::ld;
#define R1(E) [](ld x, ld, ld) -> ld { return E; }
#define R2(E) [](ld x, ld y, ld) -> ld { return E; }
#define R3(E) [](ld x, ld y, ld z) -> ld { return E; }

MC_FN(Fmod, float, fmodf(x, y));
MC_FN(Remainder, float, remainderf(x, y));
MC_FN(RemquoR, float, ([&] { int q; return remquof(x, y, &q); })());
// The low 3 bits of the quotient with its sign (all that C and CUDA guarantee).
MC_FN(RemquoQ, float, ([&] { int q; float r = remquof(x, y, &q); return isnan(r) ? 0.0f : float(q < 0 ? -(-q & 7) : (q & 7)); })());
MC_FN(FrexpM, float, ([&] { int e; return frexpf(x, &e); })());
MC_FN(FrexpE, float, ([&] { int e = 0; frexpf(x, &e); return isfinite(x) ? float(e) : 0.0f; })());
MC_FN(Ldexp, float, ldexpf(x, int(y)));
MC_FN(ModfF, float, ([&] { float ip; return modff(x, &ip); })());
MC_FN(ModfI, float, ([&] { float ip; modff(x, &ip); return ip; })());
MC_FN(Scalbn, float, scalbnf(x, int(y)));
MC_FN(Ilogb, float, float(ilogbf(x)));
MC_FN(Logb, float, logbf(x));
MC_FN(Nextafter, float, nextafterf(x, y));
MC_FN(Copysign, float, copysignf(x, y));
MC_FN(Fdim, float, fdimf(x, y));
MC_FN(Fma, float, fmaf(x, y, z));
MC_FN(Fmin, float, fminf(x, y));
MC_FN(Fmax, float, fmaxf(x, y));
MC_FN(Ceil, float, ceilf(x));
MC_FN(Floor, float, floorf(x));
MC_FN(Trunc, float, truncf(x));
MC_FN(Round, float, roundf(x));
MC_FN(Rint, float, rintf(x));
MC_FN(Classify, float, float(isnan(x) * 1 + isinf(x) * 2 + isfinite(x) * 4 + signbit(x) * 8 + (x == y) * 16 + (x < y) * 32));

static ld remquo_q(ld x, ld y) {
    int q;
    ld r = std::remquo((float)x, (float)y, &q);
    return std::isnan(r) ? 0 : (ld)(q < 0 ? -(-q & 7) : (q & 7));
}
static ld frexp_m(ld x) {
    int e;
    return std::frexp((float)x, &e);
}
static ld frexp_e(ld x) {
    int e = 0;
    std::frexp((float)x, &e);
    return std::isfinite(x) ? e : 0;
}
static ld modf_f(ld x) {
    float ip;
    return std::modf((float)x, &ip);
}
static ld modf_i(ld x) {
    float ip;
    std::modf((float)x, &ip);
    return ip;
}
// CUDA documents ilogbf(0) = ilogbf(NaN) = INT_MIN and ilogbf(inf) = INT_MAX.
static ld ilogb_ref(ld x) {
    if (x == 0 || std::isnan(x)) return (float)INT_MIN;
    if (std::isinf(x)) return (float)INT_MAX;
    return std::ilogb((float)x);
}

int main() {
    using namespace mc;
    const int n = 1 << 16;
    const float MX = FLT_MAX;
    auto big = [&](uint32_t s) { return sweep<float>(n, -1e3f, 1e3f, s, true, MX); };
    {
        std::vector<float> x, y;
        pairs(big(1), big(2), x, y);
        run<float>("fmodf", Fmod{}, R2(std::fmod(x, y)), {0}, x, y);
        run<float>("remainderf", Remainder{}, R2(std::remainder(x, y)), {0}, x, y);
        run<float>("remquof_rem", RemquoR{}, R2(std::remainder(x, y)), {0}, x, y);
        run<float>("remquof_quo", RemquoQ{}, R2(remquo_q(x, y)), {0}, x, y);
        run<float>("fdimf", Fdim{}, R2(std::fdim(x, y)), {0}, x, y);
        run<float>("copysignf", Copysign{}, R2(std::copysign(x, y)), {0}, x, y);
        run<float>("nextafterf", Nextafter{}, R2(std::nextafter((float)x, (float)y)), {0}, x, y);
        run<float>("classify_isnan_isinf_isfinite_signbit", Classify{},
                   R2(std::isnan(x) * 1 + std::isinf(x) * 2 + std::isfinite(x) * 4 + std::signbit(x) * 8 + (x == y) * 16 + (x < y) * 32), {0}, x, y);
    }
    {
        // fmin/fmax: (+0, -0) pairs are unspecified in C; keep the zeros with one sign.
        std::vector<float> x, y;
        pairs(big(3), big(4), x, y);
        for (size_t i = 0; i < x.size(); ++i)
            if (x[i] == 0 && y[i] == 0) y[i] = x[i];
        run<float>("fminf", Fmin{}, R2(std::fmin(x, y)), {0}, x, y);
        run<float>("fmaxf", Fmax{}, R2(std::fmax(x, y)), {0}, x, y);
    }
    {
        auto x = big(5);
        run<float>("frexpf_mantissa", FrexpM{}, R1(frexp_m(x)), {0}, x);
        run<float>("frexpf_exponent", FrexpE{}, R1(frexp_e(x)), {0}, x);
        run<float>("modff_frac", ModfF{}, R1(modf_f(x)), {0}, x);
        run<float>("modff_int", ModfI{}, R1(modf_i(x)), {0}, x);
        run<float>("ilogbf", Ilogb{}, R1(ilogb_ref(x)), {0}, x);
        run<float>("logbf", Logb{}, R1(std::logb((float)x)), {0}, x);
        // Rounding functions: many values near integers and halves.
        auto r = sweep<float>(n, -100, 100, 6, true, MX);
        auto u = stress::random_u32(n, 7);
        for (int i = 0; i < n; i += 2) r[i] = float(int(u[i] % 2001) - 1000) * 0.5f + ((u[i] >> 20) % 3 == 0 ? 0.0f : ((u[i] >> 20) % 3 == 1 ? 1e-3f : -1e-3f));
        run<float>("ceilf", Ceil{}, R1(std::ceil(x)), {0}, r);
        run<float>("floorf", Floor{}, R1(std::floor(x)), {0}, r);
        run<float>("truncf", Trunc{}, R1(std::trunc(x)), {0}, r);
        run<float>("roundf", Round{}, R1(std::round(x)), {0}, r);
        run<float>("rintf", Rint{}, R1(std::nearbyint(x)), {0}, r);
    }
    {
        // ldexp/scalbn: exponents in [-300, 300] reach overflow and denormal results.
        auto x = big(8);
        auto u = stress::random_u32(x.size(), 9);
        std::vector<float> e(x.size());
        for (size_t i = 0; i < x.size(); ++i) e[i] = float(int(u[i] % 601) - 300);
        run<float>("ldexpf", Ldexp{}, R2(std::ldexp(x, int(y))), {0}, x, e);
        run<float>("scalbnf", Scalbn{}, R2(std::scalbn(x, int(y))), {0}, x, e);
    }
    {
        // fmaf: one rounding of the exact x*y+z (long double holds x*y exactly).
        auto a = sweep<float>(n, -10, 10, 10, true, MX), b = sweep<float>(n, -10, 10, 11, true, MX), c = sweep<float>(n, -100, 100, 12, true, MX);
        // Cancellation cases: z close to -x*y.
        for (int i = 1; i < n; i += 4) c[i] = -(a[i] * b[i]);
        run<float>("fmaf", Fma{}, R3(std::fma(x, y, z)), {0}, a, b, c);
    }
    return stress::finish();
}
