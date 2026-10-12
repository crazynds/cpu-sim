#pragma once
// Shared checker of the math_*.cu stress programs. A function is evaluated on
// the device over a sweep of inputs (uniform over its domain, log-uniform in
// magnitude, plus special values: +-0, denormals, +-inf, NaN, huge) and
// compared with a host reference computed in long double (x87, 64-bit
// mantissa). Error bounds are the maximum ulp errors documented by NVIDIA (CUDA
// C++ Programming Guide, "Mathematical Functions"), measured against the
// long double value; an optional absolute bound covers functions documented
// with one (j0/y0 for large arguments, fast intrinsics). NaN, infinities and
// exact zeros (with their sign) must match exactly.
#include <cfloat>
#include <climits>

#include "stress.cuh"

namespace mc {

using ld = long double;
using Ref1 = ld (*)(ld);
using Ref2 = ld (*)(ld, ld);
using Ref3 = ld (*)(ld, ld, ld);

template <class T>
constexpr int digits() {
    return std::numeric_limits<T>::digits;
}

// ulp of the T value nearest to r.
template <class T>
ld ulp_of(ld r) {
    if (r == 0 || !std::isfinite(r)) return (ld)std::numeric_limits<T>::denorm_min();
    int e;
    std::frexp(r, &e);  // r = m * 2^e, 0.5 <= |m| < 1
    ld u = std::ldexp((ld)1, e - digits<T>());
    return std::max(u, (ld)std::numeric_limits<T>::denorm_min());
}

struct Bound {
    double ulps;     // max ulp error (0 = correctly rounded, checked as <= 0.5 ulp)
    double abs = 0;  // alternatively, max absolute error
    // Input-dependent bounds (fast intrinsics): overrides ulps/abs when set.
    void (*fn)(ld x, ld y, double& ulps, double& abs) = nullptr;
};

// Check one result against the long double reference. Returns the error in ulps
// (or -1 on a class mismatch).
template <class T>
bool accept(T got, ld ref, Bound b, double* err) {
    *err = 0;
    if (std::isnan(ref)) return std::isnan(got);
    if (std::isinf(ref)) return got == (T)ref;
    if (ref == 0) return got == 0 && std::signbit(got) == std::signbit(ref);
    if (std::isnan(got)) return false;
    T r = (T)ref;
    if (std::isinf(got)) {
        // Fine if a value within the bound of the exact one rounds to infinity.
        const ld mx = (ld)std::numeric_limits<T>::max(), thr = mx + ulp_of<T>(mx) / 2;
        return std::signbit(got) == std::signbit(ref) && std::fabs(ref) + std::max(b.ulps, 0.5) * ulp_of<T>(ref) >= thr;
    }
    ld e = std::fabs((ld)got - ref) / ulp_of<T>(ref);
    *err = (double)e;
    double lim = std::max(b.ulps, 0.5) + 1e-3;
    return e <= lim || (b.abs > 0 && std::fabs((ld)got - ref) <= b.abs);
}

// Inputs: half uniform in [lo, hi], half log-uniform in magnitude within the
// domain (signs as the domain allows), then the special values.
// logmax (when nonzero) extends the log-uniform half up to that magnitude.
template <class T>
std::vector<T> sweep(int n, T lo, T hi, uint32_t seed, bool specials = true, T logmax = 0) {
    std::vector<T> v;
    v.reserve(n + 64);
    auto r = stress::random_u32(2 * size_t(n), seed);
    T amax = logmax != 0 ? logmax : std::max(std::fabs(lo), std::fabs(hi));
    int emax = std::isfinite(amax) ? std::ilogb(amax) : std::numeric_limits<T>::max_exponent - 1;
    int emin = std::numeric_limits<T>::min_exponent - 1 - 4;  // a few denormal binades
    for (int i = 0; i < n; ++i) {
        double u = (r[2 * i] >> 8) / 16777216.0 + (r[2 * i + 1] >> 8) / 16777216.0 / 16777216.0;
        T x;
        if (i % 2 == 0 && std::isfinite(hi - lo)) {
            x = (T)(lo + (hi - lo) * u);
        } else {
            int e = emin + int(r[2 * i + 1] % unsigned(emax - emin + 1));
            T m = (T)(1.0 + u);
            x = std::ldexp(m, e);
            bool neg = lo < 0 && (hi <= 0 || (r[2 * i] & 1));
            if (neg) x = -x;
            if (logmax == 0 && (x < lo || x > hi)) x = (T)(lo + (hi - lo) * u);
            if (logmax != 0 && lo >= 0 && x < lo) x = (T)(lo + (hi - lo) * u);
            if (!std::isfinite(x)) x = lo;
        }
        v.push_back(x);
    }
    if (specials) {
        const T inf = std::numeric_limits<T>::infinity(), dm = std::numeric_limits<T>::denorm_min(), mn = std::numeric_limits<T>::min(),
                mx = std::numeric_limits<T>::max();
        const T sp[] = {T(0), -T(0), T(1), T(-1), T(0.5), T(-0.5), T(2), T(-2), dm, -dm, mn, -mn, mn / 4, -mn / 3, mx, -mx, inf, -inf,
                        std::numeric_limits<T>::quiet_NaN(), T(1e30), T(-1e30), T(100), T(-100), T(3.14159265358979), T(1e-7),
                        T(0.999999), T(1.000001), T(10), T(-10), T(1e-30)};
        v.insert(v.end(), std::begin(sp), std::end(sp));
    }
    return v;
}

template <class T>
inline void print_mismatch(const char* name, size_t i, const T* args, int nargs, T got, ld ref, double err) {
    std::printf("  %s mismatch [%zu] args=(", name, i);
    for (int k = 0; k < nargs; ++k) std::printf("%s%.9g", k ? ", " : "", double(args[k]));
    std::printf("): got %.17g want %.17Lg (%.2f ulp)\n", double(got), ref, err);
}

// One device evaluation per element, generic over arity through a functor F
// that takes (x, y, z) and ignores what it does not use.
template <class T, class F>
__global__ void k_eval(F f, const T* x, const T* y, const T* z, T* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f(x[i], y[i], z[i]);
}

// Runs one case: inputs x (and y, z when given), reference r(x, y, z).
template <class T, class F, class R>
void run(const char* name, F f, R ref, Bound b, std::vector<T> x, std::vector<T> y = {}, std::vector<T> z = {}, int nargs = 0) {
    using namespace stress;
    size_t n = x.size();
    if (!nargs) nargs = z.size() ? 3 : y.size() ? 2 : 1;
    if (y.size() < n) y.resize(n, T(1));
    if (z.size() < n) z.resize(n, T(1));
    std::vector<ld> want(n);
    double rt = best_ms([&] {
#pragma omp parallel for
        for (long long i = 0; i < (long long)n; ++i) want[i] = ref((ld)x[i], (ld)y[i], (ld)z[i]);
    });
    Buffer<T> dx(x), dy(y), dz(z), dout(n);
    double kt = time_device([&] { k_eval<T><<<unsigned((n + 255) / 256), 256>>>(f, dx.p, dy.p, dz.p, dout.p, int(n)); });
    auto got = dout.download();
    bool ok = true;
    double worst = 0;
    for (size_t i = 0; i < n; ++i) {
        double e;
        Bound bi = b;
        if (b.fn) b.fn((ld)x[i], (ld)y[i], bi.ulps, bi.abs);
        if (!accept(got[i], want[i], bi, &e)) {
            T args[3] = {x[i], y[i], z[i]};
            if (ok) print_mismatch(name, i, args, nargs, got[i], want[i], e);
            ok = false;
        }
        worst = std::max(worst, e);
    }
    char note[64];
    std::snprintf(note, sizeof note, "max_ulp=%.2f bound=%.1f", worst, b.ulps);
    report(name, ok, kt, rt, note);
}

// Cartesian-ish pairs: every x with a shuffled y, plus all specials x specials.
template <class T>
void pairs(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& x, std::vector<T>& y, int nspecial = 30) {
    x = a, y.resize(a.size());
    for (size_t i = 0; i < a.size(); ++i) y[i] = b[(i * 7919 + 13) % b.size()];
    size_t sa = a.size() - nspecial, sb = b.size() - nspecial;
    for (int i = 0; i < nspecial; ++i)
        for (int j = 0; j < nspecial; ++j) x.push_back(a[sa + i]), y.push_back(b[sb + j]);
}

}  // namespace mc

// Functor definitions: a struct with a __device__ call operator over (x, y, z).
#define MC_FN(NAME, T, EXPR)                                                    \
    struct NAME {                                                                \
        __device__ T operator()(T x, T y, T z) const {                           \
            (void)y, (void)z;                                                    \
            return EXPR;                                                         \
        }                                                                        \
    }
