// C = A * B, 4100 x 4130 x 4070 (bench/matmul.cu). Every version adds the
// products of each C[i][j] in increasing k, as the CUDA kernels do.
//   matmul [ijk-rows]   (ijk-rows > 0 also times the CUDA-like ijk loop on that many rows)
#include <cstdlib>

#include "native.hpp"

constexpr int M = 4100, N = 4070, K = 4130;

// The naive kernel's loop as is: one output element at a time, B read down a
// column (stride N). Not vectorizable; only timed on a few rows.
static void ijk(const float* A, const float* B, float* C, int rows) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += A[size_t(i) * K + k] * B[size_t(k) * N + j];
            C[size_t(i) * N + j] = acc;
        }
}

// Same per-element order with the loops swapped: the inner loop walks a row
// of B and of C, contiguous, and vectorizes.
static void ikj(const float* A, const float* B, float* C) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        float* c = C + size_t(i) * N;
        std::memset(c, 0, sizeof(float) * N);
        for (int k = 0; k < K; ++k) {
            const float a = A[size_t(i) * K + k];
            const float* b = B + size_t(k) * N;
            for (int j = 0; j < N; ++j) c[j] += a * b[j];
        }
    }
}

// Cache blocking (what the tiled CUDA kernel does with shared memory): a block
// of C stays in cache while k advances, still in increasing k per element.
static void blocked(const float* A, const float* B, float* C) {
    constexpr int BI = 64, BJ = 512, BK = 256;
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i0 = 0; i0 < M; i0 += BI)
        for (int j0 = 0; j0 < N; j0 += BJ) {
            const int i1 = i0 + BI < M ? i0 + BI : M, j1 = j0 + BJ < N ? j0 + BJ : N;
            for (int i = i0; i < i1; ++i) std::memset(C + size_t(i) * N + j0, 0, sizeof(float) * (j1 - j0));
            for (int k0 = 0; k0 < K; k0 += BK) {
                const int k1 = k0 + BK < K ? k0 + BK : K;
                for (int i = i0; i < i1; ++i) {
                    float* c = C + size_t(i) * N;
                    for (int k = k0; k < k1; ++k) {
                        const float a = A[size_t(i) * K + k];
                        const float* b = B + size_t(k) * N;
                        for (int j = j0; j < j1; ++j) c[j] += a * b[j];
                    }
                }
            }
        }
}

int main(int argc, char** argv) {
    auto A = random_floats(size_t(M) * K, 1, -1, 1), B = random_floats(size_t(K) * N, 2, -1, 1);
    std::vector<float> C(size_t(M) * N);
    if (int rows = argc > 1 ? std::atoi(argv[1]) : 0; rows > 0) {
        double ms = native_ms([&] { ijk(A.data(), B.data(), C.data(), rows); }, 1);
        std::printf("NATIVE %-22s %10.3f ms  threads %d  (%d of %d rows; whole matrix ~%.0f ms)\n", "matmul_ijk", ms,
                    omp_get_max_threads(), rows, M, ms * M / rows);
    }
    native_report("matmul_ikj", native_ms([&] { ikj(A.data(), B.data(), C.data()); }));
    native_checksum("matmul_naive", C);
    native_report("matmul_blocked", native_ms([&] { blocked(A.data(), B.data(), C.data()); }));
    native_checksum("matmul_tiled", C);
}
