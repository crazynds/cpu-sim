#pragma once
// Plain C++ versions of the benchmarked kernels, to compare the CUDA code run
// by our executor with code written directly for the CPU. Same data (same
// generator and seeds as bench/*.cu) and same results bit for bit, so the
// CHECK hashes match the GPU's. Parallel loops use OpenMP: OMP_NUM_THREADS=1
// is the single-core version. Built with -O3 -march=native -ffp-contract=off
// (no fused multiply-add, like the samples' -fmad=false).
#include <omp.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct Lcg {  // same generator as samples/common/common.cuh
    uint64_t state;
    explicit Lcg(uint64_t seed) : state(seed) {}
    uint32_t next_u32() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(state >> 33);
    }
    float next_float(float lo, float hi) { return lo + (hi - lo) * (static_cast<float>(next_u32()) / 2147483648.0f); }
};

inline std::vector<float> random_floats(size_t n, uint64_t seed, float lo, float hi) {
    Lcg rng(seed);
    std::vector<float> v(n);
    for (auto& x : v) x = rng.next_float(lo, hi);
    return v;
}

// Best of `reps` runs after one warm-up; `setup` (untimed) runs before each.
template <typename F, typename S>
double native_ms(F&& run, S&& setup, int reps = 3) {
    setup();
    run();
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        setup();
        auto t0 = std::chrono::steady_clock::now();
        run();
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        best = ms < best ? ms : best;
    }
    return best;
}
template <typename F>
double native_ms(F&& run, int reps = 3) {
    return native_ms(run, [] {}, reps);
}

inline void native_report(const char* name, double ms) {
    std::printf("NATIVE %-22s %10.3f ms  threads %d\n", name, ms, omp_get_max_threads());
}

// Named like the GPU kernel whose result it reproduces (bench/run.sh compares).
template <typename T>
void native_checksum(const char* gpu_kernel, const std::vector<T>& v) {
    uint64_t h = 1469598103934665603ull;
    const auto* p = reinterpret_cast<const unsigned char*>(v.data());
    for (size_t i = 0; i < v.size() * sizeof(T); ++i) h = (h ^ p[i]) * 1099511628211ull;
    std::printf("CHECK %-22s %016llx\n", gpu_kernel, (unsigned long long)h);
}
