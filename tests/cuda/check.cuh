#pragma once
// Helpers for the self-checking CUDA feature tests (tests/cuda_feature_test.sh).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                                              \
    do {                                                                                              \
        cudaError_t e_ = (call);                                                                      \
        if (e_ != cudaSuccess) {                                                                      \
            std::printf("%s:%d: %s -> %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_));      \
            std::exit(1);                                                                             \
        }                                                                                             \
    } while (0)

// Counts failures, printing the first few.
struct Checker {
    const char* name;
    int failures = 0;
    explicit Checker(const char* n) : name(n) {}
    void expect(bool ok, const char* what, long long index = -1, double got = 0, double want = 0) {
        if (ok) return;
        if (failures++ < 10) std::printf("  %s: %s [%lld] got %.17g want %.17g\n", name, what, index, got, want);
    }
    int finish() {
        std::printf("%s: %s\n", name, failures ? "FAIL" : "PASS");
        return failures ? 1 : 0;
    }
};

inline bool close_rel(double got, double want, double rel, double abs_tol = 0) {
    if (std::isnan(want)) return std::isnan(got);
    if (std::isinf(want)) return got == want;
    return std::fabs(got - want) <= abs_tol + rel * std::fabs(want);
}
