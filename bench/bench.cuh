#pragma once
// Kernel-only benchmarks. Each bench includes a sample's .cu with its main()
// renamed, so the kernels are exactly the sample's, and times only the kernel
// launches (no reference computation, no copies): best of `reps` runs.
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

// CPU cycles of this process and every thread it creates afterwards (the
// simulator's workers): a measure of work that does not depend on the clock
// frequency, which on a laptop moves with temperature. -1 if unavailable.
inline long long process_cycles() {
    static int fd = [] {
        perf_event_attr attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.size = sizeof(attr);
        attr.type = PERF_TYPE_HARDWARE;
        attr.config = PERF_COUNT_HW_CPU_CYCLES;
        attr.inherit = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        return int(syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0));
    }();
    long long v = -1;
    if (fd < 0 || read(fd, &v, sizeof(v)) != sizeof(v)) return -1;
    return v;
}

struct BenchResult {
    double ms;
    double gcycles;  // CPU cycles spent during the run, in billions (host side for the GPU build)
};

// Best of `reps` runs (by time), after one warm-up run.
template <typename F>
BenchResult bench_ms(F&& launch, int reps = 5) {
    process_cycles();  // open the counter before the workers exist
    launch();
    CUDA_CHECK_KERNEL();
    BenchResult best{1e30, 0};
    for (int i = 0; i < reps; ++i) {
        long long c0 = process_cycles();
        auto t0 = std::chrono::steady_clock::now();
        launch();
        CUDA_CHECK_KERNEL();
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        long long c1 = process_cycles();
        if (ms < best.ms) best = {ms, c0 >= 0 ? (c1 - c0) / 1e9 : -1};
    }
    return best;
}

inline void bench_report(const char* name, BenchResult r) {
    std::printf("BENCH %-22s %10.3f ms %9.2f Gcyc\n", name, r.ms, r.gcycles);
}

template <typename T>
std::vector<T> random_floats(size_t n, uint64_t seed, float lo, float hi) {
    Lcg rng(seed);
    std::vector<T> v(n);
    for (auto& x : v) x = rng.next_float(lo, hi);
    return v;
}

// Hash of a device buffer's contents. With -fmad=false the samples' results are
// bit-identical on the GPU and on the CPU, so run.sh compares these.
template <typename T>
void bench_checksum(const char* name, const DeviceBuffer<T>& buf) {
    std::vector<T> host;
    buf.download(host);
    uint64_t h = 1469598103934665603ull;
    const auto* p = reinterpret_cast<const unsigned char*>(host.data());
    for (size_t i = 0; i < host.size() * sizeof(T); ++i) h = (h ^ p[i]) * 1099511628211ull;
    std::printf("CHECK %-22s %016llx\n", name, (unsigned long long)h);
}
