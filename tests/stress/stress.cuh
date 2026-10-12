#pragma once
// Harness of the stress suite (tests/stress/run.sh). Each program checks a
// group of cases against a host reference and prints one line per case:
//
//   CASE <name> PASS|FAIL kernel_ms=<t> ref_ms=<t> [note]
//
// kernel_ms: the device work of the case (best of a few runs after a
// warm-up); ref_ms: the same computation on the host with OpenMP, every
// thread (-1 when the case has no meaningful host equivalent). The ratio is
// what the suite uses to find slow cases, so the two must compute the same
// thing on the same data.
//
// The same source builds with nvcc (the GPU is the oracle: every case must
// pass there) and with cucc.
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define STRESS_CHECK(call)                                                                              \
    do {                                                                                                \
        cudaError_t e_ = (call);                                                                        \
        if (e_ != cudaSuccess) {                                                                        \
            std::printf("ERROR %s:%d: %s -> %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_));  \
            std::fflush(stdout);                                                                        \
            std::exit(2);                                                                               \
        }                                                                                               \
    } while (0)

namespace stress {

inline double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Best of `reps` timed runs of f after one untimed run. f must finish its
// device work (call sync() at the end of a device run).
inline double best_ms(const std::function<void()>& f, int reps = 3) {
    f();
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        double t0 = now_ms();
        f();
        best = std::min(best, now_ms() - t0);
    }
    return best;
}

inline void sync() { STRESS_CHECK(cudaDeviceSynchronize()); }

// One kernel launch (or a few) timed: the lambda launches, this synchronizes.
inline double time_device(const std::function<void()>& launch, int reps = 3) {
    return best_ms([&] {
        launch();
        STRESS_CHECK(cudaGetLastError());
        sync();
    }, reps);
}

inline int failures = 0;

inline void report(const char* name, bool ok, double kernel_ms, double ref_ms, const char* note = "") {
    if (!ok) ++failures;
    std::printf("CASE %s %s kernel_ms=%.4f ref_ms=%.4f%s%s\n", name, ok ? "PASS" : "FAIL", kernel_ms, ref_ms,
                note[0] ? " " : "", note);
    std::fflush(stdout);
}

// Comparisons that print the first mismatch.
template <class T>
bool equal(const T* got, const T* want, size_t n, const char* what = "") {
    for (size_t i = 0; i < n; ++i)
        if (!(got[i] == want[i])) {
            std::printf("  mismatch %s[%zu]: got %.17g want %.17g\n", what, i, double(got[i]), double(want[i]));
            return false;
        }
    return true;
}

template <class T>
bool close(const T* got, const T* want, size_t n, double rel, double abs_tol = 0, const char* what = "") {
    for (size_t i = 0; i < n; ++i) {
        double g = double(got[i]), w = double(want[i]);
        bool ok = std::isnan(w) ? std::isnan(g) : std::isinf(w) ? g == w : std::fabs(g - w) <= abs_tol + rel * std::fabs(w);
        if (!ok) {
            std::printf("  mismatch %s[%zu]: got %.17g want %.17g\n", what, i, g, w);
            return false;
        }
    }
    return true;
}

// Deterministic data.
inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}
inline std::vector<float> random_floats(size_t n, uint32_t seed, float lo = -1, float hi = 1) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = lo + (hi - lo) * float(hash32(uint32_t(i) * 2654435761u + seed) >> 8) / 16777216.0f;
    return v;
}
inline std::vector<uint32_t> random_u32(size_t n, uint32_t seed) {
    std::vector<uint32_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = hash32(uint32_t(i) * 2654435761u + seed);
    return v;
}

// Device buffer with copies in and out.
template <class T>
struct Buffer {
    T* p = nullptr;
    size_t n = 0;
    explicit Buffer(size_t count) : n(count) { STRESS_CHECK(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T))); }
    explicit Buffer(const std::vector<T>& v) : Buffer(v.size()) { upload(v); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<T>& v) { STRESS_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> download() const {
        std::vector<T> v(n);
        STRESS_CHECK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
    void zero() { STRESS_CHECK(cudaMemset(p, 0, n * sizeof(T))); }
    operator T*() const { return p; }
};

inline int finish() {
    std::printf("DONE failures=%d\n", failures);
    return failures ? 1 : 0;
}

}  // namespace stress
