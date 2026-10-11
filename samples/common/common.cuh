#pragma once
// Shared helpers for the CudaSimulator sample programs.
//
// The samples intentionally use only a small subset of the CUDA runtime API,
// which is the first surface our own runtime will have to provide:
//   cudaMalloc, cudaFree, cudaMemcpy, cudaMemset, cudaDeviceSynchronize,
//   cudaGetLastError, cudaGetErrorString

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

// Problem sizes: GPU-sized by default (>= 100 ms on an RTX 5060 where the
// kernel allows it), smaller with -DSAMPLE_TEST, which the ctest runs use:
// the same code paths and edge cases in seconds instead of minutes.
#ifdef SAMPLE_TEST
#define SAMPLE_SIZE(full, test) (test)
#else
#define SAMPLE_SIZE(full, test) (full)
#endif

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__,   \
                         __LINE__, cudaGetErrorString(err_), #call);           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)

// Deterministic generator so the GPU run and the CPU simulator run see the
// exact same inputs.
struct Lcg {
    uint64_t state;

    explicit Lcg(uint64_t seed) : state(seed) {}

    uint32_t next_u32() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(state >> 33);  // 31 random bits
    }

    float next_float(float lo, float hi) {
        return lo + (hi - lo) * (static_cast<float>(next_u32()) / 2147483648.0f);
    }
};

// RAII wrapper over a device allocation.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(ptr_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }
    size_t size() const { return count_; }

    void upload(const std::vector<T>& host) {
        CUDA_CHECK(cudaMemcpy(ptr_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(std::vector<T>& host) const {
        host.resize(count_);
        CUDA_CHECK(cudaMemcpy(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void zero() { CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T))); }

private:
    T* ptr_ = nullptr;
    size_t count_;
};

// Runs `launch` once as warm-up and once timed. `launch` must be idempotent
// (reset its own outputs) because it runs twice.
template <typename F>
double time_ms(F&& launch) {
    launch();
    CUDA_CHECK_KERNEL();
    auto start = std::chrono::steady_clock::now();
    launch();
    CUDA_CHECK_KERNEL();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

inline void report_time(const char* what, double ms) {
    std::printf("  %-28s %10.3f ms\n", what, ms);
}

// |ref - got| <= abs_tol + rel_tol * |ref| for every element (NaN fails).
inline bool check_close(const float* ref, const float* got, size_t n, double rel_tol,
                        double abs_tol, const char* what) {
    size_t bad = 0, first_bad = n;
    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double r = ref[i], g = got[i];
        double err = std::fabs(r - g);
        if (!(err <= abs_tol + rel_tol * std::fabs(r))) {
            if (first_bad == n) first_bad = i;
            ++bad;
        }
        if (err > max_err) max_err = err;
    }
    std::printf("  %-28s max abs err %.3g, %zu/%zu mismatches\n", what, max_err, bad, n);
    if (bad != 0)
        std::printf("    first mismatch at %zu: expected %.9g, got %.9g\n", first_bad,
                    static_cast<double>(ref[first_bad]), static_cast<double>(got[first_bad]));
    return bad == 0;
}

template <typename T>
bool check_equal(const T* ref, const T* got, size_t n, const char* what) {
    static_assert(std::is_integral_v<T>, "check_equal is for integer data");
    size_t bad = 0, first_bad = n;
    for (size_t i = 0; i < n; ++i) {
        if (ref[i] != got[i]) {
            if (first_bad == n) first_bad = i;
            ++bad;
        }
    }
    std::printf("  %-28s %zu/%zu mismatches\n", what, bad, n);
    if (bad != 0)
        std::printf("    first mismatch at %zu: expected %lld, got %lld\n", first_bad,
                    static_cast<long long>(ref[first_bad]), static_cast<long long>(got[first_bad]));
    return bad == 0;
}

inline int finish(const char* sample, bool ok) {
    std::printf("%s: %s\n", sample, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
