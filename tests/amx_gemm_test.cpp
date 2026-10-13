// The AMX GEMM (src/device/amx_gemm.cpp) against a plain reference: bf16 and
// int8, C = A*B and C += A*B, leading dimensions larger than the matrices,
// A read in place (rows 64-byte aligned) and packed (rows that are not).
// On a CPU without AMX it reports SKIP. With an argument, also the speed of
// a square product of that size:  test_amx_gemm [n]
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "common/topology.h"
#include "device/amx_gemm.h"

using namespace cpusim;

namespace {

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
float from_bf16(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

int failures = 0;

// 64-byte aligned vectors, like device memory.
template <typename T>
struct Aligned {
    using value_type = T;
    Aligned() = default;
    template <typename U>
    Aligned(const Aligned<U>&) {}
    T* allocate(size_t n) { return static_cast<T*>(std::aligned_alloc(64, (n * sizeof(T) + 63) / 64 * 64)); }
    void deallocate(T* p, size_t) { std::free(p); }
    bool operator==(const Aligned&) const { return true; }
};
template <typename T>
using Vec = std::vector<T, Aligned<T>>;

template <bool INT8>
void check(Executor& ex, uint32_t M, uint32_t N, uint32_t K, uint32_t pad, bool accumulate) {
    using T = std::conditional_t<INT8, int8_t, uint16_t>;
    using Acc = std::conditional_t<INT8, int32_t, float>;
    const uint64_t lda = K + pad, ldb = N + pad, ldc = N + pad;
    Vec<T> A(M * lda), B(K * ldb);
    Vec<Acc> C(M * ldc), want(M * ldc);
    uint32_t x = 12345;
    auto rnd = [&] { return x = x * 1664525u + 1013904223u, int(x >> 24) % 9 - 4; };  // small integers: sums exact
    for (auto& v : A) v = INT8 ? T(rnd()) : T(to_bf16(float(rnd())));
    for (auto& v : B) v = INT8 ? T(rnd()) : T(to_bf16(float(rnd())));
    for (size_t i = 0; i < C.size(); ++i) C[i] = want[i] = Acc(rnd());
    for (uint32_t i = 0; i < M; ++i)
        for (uint32_t j = 0; j < N; ++j) {
            double s = accumulate ? double(want[i * ldc + j]) : 0;
            for (uint32_t k = 0; k < K; ++k) {
                const double a = INT8 ? double(A[i * lda + k]) : double(from_bf16(uint16_t(A[i * lda + k])));
                const double b = INT8 ? double(B[k * ldb + j]) : double(from_bf16(uint16_t(B[k * ldb + j])));
                s += a * b;
            }
            want[i * ldc + j] = Acc(s);
        }
    GemmArgs g;
    g.type = INT8 ? GemmType::S8_S32 : GemmType::BF16_F32;
    g.A = A.data(), g.B = B.data(), g.C = C.data();
    g.lda = lda, g.ldb = ldb, g.ldc = ldc;
    g.M = M, g.N = N, g.K = K, g.accumulate = accumulate;
    if (!amx_gemm_supported(g)) {
        std::printf("  %s %ux%ux%u: not supported\n", INT8 ? "int8" : "bf16", M, N, K);
        ++failures;
        return;
    }
    KernelFault f = amx_gemm(ex, g);
    size_t bad = 0;
    for (uint32_t i = 0; i < M; ++i)
        for (uint32_t j = 0; j < N; ++j) bad += C[i * ldc + j] != want[i * ldc + j];
    std::printf("  %s %ux%ux%u ld+%u %s: %s\n", INT8 ? "int8" : "bf16", M, N, K, pad, accumulate ? "C+=AB" : "C=AB",
                f.signal || bad ? "FAIL" : "ok");
    failures += f.signal || bad;
}

template <bool INT8>
void speed(Executor& ex, uint32_t n) {
    using T = std::conditional_t<INT8, int8_t, uint16_t>;
    using Acc = std::conditional_t<INT8, int32_t, float>;
    Vec<T> A(size_t(n) * n, T(INT8 ? 1 : 0x3f80)), B(size_t(n) * n, T(INT8 ? 1 : 0x3f80));
    Vec<Acc> C(size_t(n) * n);
    GemmArgs g;
    g.type = INT8 ? GemmType::S8_S32 : GemmType::BF16_F32;
    g.A = A.data(), g.B = B.data(), g.C = C.data();
    g.lda = g.ldb = g.ldc = n;
    g.M = g.N = g.K = n;
    double best = 1e30;
    for (int r = 0; r < 6; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        amx_gemm(ex, g);
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::printf("  %s %ux%ux%u: %.2f ms, %.1f T%s (packing included)\n", INT8 ? "int8" : "bf16", n, n, n, best * 1e3,
                2.0 * n * n * double(n) / best / 1e12, INT8 ? "OPS" : "FLOPS");
}

}  // namespace

int main(int argc, char** argv) {
    if (!amx_gemm_available(GemmType::BF16_F32) || !amx_gemm_available(GemmType::S8_S32)) {
        std::printf("amx_gemm: SKIP (no AMX)\n");
        return 0;
    }
    std::vector<int> cpus;
    for (const auto& core : read_topology().cores)
        for (int c : core.cpus) cpus.push_back(c);
    Executor ex(cpus);
    for (bool acc : {false, true}) {
        check<false>(ex, 64, 64, 64, 0, acc);
        check<false>(ex, 96, 160, 224, 8, acc);
        check<true>(ex, 64, 64, 64, 0, acc);
        check<true>(ex, 160, 96, 192, 16, acc);
    }
    check<false>(ex, 512, 480, 1024, 0, false);
    check<true>(ex, 480, 512, 1024, 0, false);
    if (argc > 1) {
        speed<false>(ex, uint32_t(std::atoi(argv[1])));
        speed<true>(ex, uint32_t(std::atoi(argv[1])));
    }
    std::printf("amx_gemm: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
