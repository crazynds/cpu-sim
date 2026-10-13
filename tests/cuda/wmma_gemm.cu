// GEMMs written with wmma, as the runtime recognizes them (and runs them on
// AMX tiles where the CPU has them; elsewhere the kernels run as written):
// one 16x16 tile per warp in bf16 and in int8 with bounds checks, C += A*B,
// a 2x2 block of tiles per warp, and a kernel that scales its accumulator
// (not a plain GEMM: must not be recognized, and must still be right).
// Small integers: every sum is exact, results compared exactly.
#include <cuda_bf16.h>
#include <mma.h>

#include <cstdint>
#include <type_traits>
#include <vector>

#include "check.cuh"

using namespace nvcuda;

// One tile per warp: warps along M in x, along N in y.
template <typename T, typename Acc>
__global__ void gemm_tile(const T* a, const T* b, Acc* c, int M, int N, int K) {
    const int wm = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize, wn = blockIdx.y * blockDim.y + threadIdx.y;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, Acc> acc;
    wmma::fill_fragment(acc, Acc(0));
    for (int k = 0; k < K; k += 16)
        if (wm * 16 < M && wn * 16 < N) {
            wmma::load_matrix_sync(af, a + size_t(wm) * 16 * K + k, K);
            wmma::load_matrix_sync(bf, b + size_t(k) * N + wn * 16, N);
            wmma::mma_sync(acc, af, bf, acc);
        }
    if (wm * 16 < M && wn * 16 < N) wmma::store_matrix_sync(c + size_t(wm) * 16 * N + wn * 16, acc, N, wmma::mem_row_major);
}

// C += A * B (the accumulator starts from C).
__global__ void gemm_accumulate(const __nv_bfloat16* a, const __nv_bfloat16* b, float* c, int M, int N, int K) {
    const int wm = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize, wn = blockIdx.y;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    float* ct = c + size_t(wm) * 16 * N + wn * 16;
    wmma::load_matrix_sync(acc, ct, N, wmma::mem_row_major);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(af, a + size_t(wm) * 16 * K + k, K);
        wmma::load_matrix_sync(bf, b + size_t(k) * N + wn * 16, N);
        wmma::mma_sync(acc, af, bf, acc);
    }
    wmma::store_matrix_sync(ct, acc, N, wmma::mem_row_major);
}

// A 32x32 block of C per warp (2x2 accumulators), lda/ldb/ldc as parameters.
__global__ void gemm_2x2(const __nv_bfloat16* a, const __nv_bfloat16* b, float* c, int K, int lda, int ldb, int ldc) {
    const int wm = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, wn = blockIdx.y;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    for (int k = 0; k < K; k += 16) {
        for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(af[i], a + size_t(wm * 32 + i * 16) * lda + k, lda);
        for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(bf[j], b + size_t(k) * ldb + wn * 32 + j * 16, ldb);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(c + size_t(wm * 32 + i * 16) * ldc + wn * 32 + j * 16, acc[i][j], ldc,
                                    wmma::mem_row_major);
}

// Not a plain GEMM: 2 * A * B.
__global__ void gemm_scaled(const __nv_bfloat16* a, const __nv_bfloat16* b, float* c, int M, int N, int K) {
    const int wm = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize, wn = blockIdx.y;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(af, a + size_t(wm) * 16 * K + k, K);
        wmma::load_matrix_sync(bf, b + size_t(k) * N + wn * 16, N);
        wmma::mma_sync(acc, af, bf, acc);
    }
    for (int i = 0; i < acc.num_elements; ++i) acc.x[i] *= 2.0f;
    wmma::store_matrix_sync(c + size_t(wm) * 16 * N + wn * 16, acc, N, wmma::mem_row_major);
}

template <typename T>
static T to(int v) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) return __float2bfloat16(float(v));
    else return T(v);
}

int main() {
    Checker check("wmma_gemm");
    auto ints = [](size_t n, uint32_t seed) {
        std::vector<int> v(n);
        for (auto& x : v) x = int((seed = seed * 1664525u + 1013904223u) >> 24) % 7 - 3;
        return v;
    };
    // Reference: C (ldc) = scale * A (lda) * B (ldb) [+ C0].
    auto ref = [](const std::vector<int>& A, const std::vector<int>& B, int M, int N, int K, int lda, int ldb,
                  double scale, const std::vector<double>* c0, int ldc) {
        std::vector<double> C(size_t(M) * ldc, 0);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                double s = 0;
                for (int k = 0; k < K; ++k) s += double(A[size_t(i) * lda + k]) * B[size_t(k) * ldb + j];
                C[size_t(i) * ldc + j] = scale * s + (c0 ? (*c0)[size_t(i) * ldc + j] : 0);
            }
        return C;
    };
    auto run = [&](const char* name, auto launch, int M, int N, int K, int lda, int ldb, int ldc, bool int8, double scale,
                   bool accumulate) {
        const auto A = ints(size_t(M) * lda, 1), B = ints(size_t(K) * ldb, 2), C0i = ints(size_t(M) * ldc, 3);
        std::vector<double> c0(C0i.begin(), C0i.end());
        const auto want = ref(A, B, M, N, K, lda, ldb, scale, accumulate ? &c0 : nullptr, ldc);
        void *da, *db, *dc;
        const size_t es = int8 ? 1 : 2;
        CHECK_CUDA(cudaMalloc(&da, A.size() * es));
        CHECK_CUDA(cudaMalloc(&db, B.size() * es));
        CHECK_CUDA(cudaMalloc(&dc, size_t(M) * ldc * 4));
        if (int8) {
            std::vector<signed char> a(A.begin(), A.end()), b(B.begin(), B.end());
            CHECK_CUDA(cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice));
            std::vector<int> c(C0i);
            CHECK_CUDA(cudaMemcpy(dc, c.data(), c.size() * 4, cudaMemcpyHostToDevice));
        } else {
            std::vector<__nv_bfloat16> a(A.size()), b(B.size());
            for (size_t i = 0; i < A.size(); ++i) a[i] = to<__nv_bfloat16>(A[i]);
            for (size_t i = 0; i < B.size(); ++i) b[i] = to<__nv_bfloat16>(B[i]);
            CHECK_CUDA(cudaMemcpy(da, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(db, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
            std::vector<float> c(C0i.begin(), C0i.end());
            CHECK_CUDA(cudaMemcpy(dc, c.data(), c.size() * 4, cudaMemcpyHostToDevice));
        }
        for (int rep = 0; rep < 2; ++rep) {  // twice: the second launch uses the cached plan
            if (accumulate && rep == 1) break;  // C += A*B would add twice
            launch(da, db, dc);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        std::vector<float> got(size_t(M) * ldc);
        if (int8) {
            std::vector<int> g(got.size());
            CHECK_CUDA(cudaMemcpy(g.data(), dc, g.size() * 4, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < g.size(); ++i) got[i] = float(g[i]);
        } else {
            CHECK_CUDA(cudaMemcpy(got.data(), dc, got.size() * 4, cudaMemcpyDeviceToHost));
        }
        int bad = 0;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                if (double(got[size_t(i) * ldc + j]) != want[size_t(i) * ldc + j]) {
                    if (bad++ < 3)
                        check.expect(false, name, long(i) * ldc + j, got[size_t(i) * ldc + j], want[size_t(i) * ldc + j]);
                }
        std::printf("  %s %dx%dx%d: %s\n", name, M, N, K, bad ? "FAIL" : "ok");
        CHECK_CUDA(cudaFree(da));
        CHECK_CUDA(cudaFree(db));
        CHECK_CUDA(cudaFree(dc));
    };
    const int M = 256, N = 192, K = 320;
    run("bf16 tile per warp", [&](void* a, void* b, void* c) {
        gemm_tile<<<dim3(M / 16 / 4, N / 16 / 2), dim3(128, 2)>>>((const __nv_bfloat16*)a, (const __nv_bfloat16*)b, (float*)c, M, N, K);
    }, M, N, K, K, N, N, false, 1.0, false);
    run("int8 tile per warp", [&](void* a, void* b, void* c) {
        gemm_tile<<<dim3(M / 16 / 4, N / 16 / 2), dim3(128, 2)>>>((const signed char*)a, (const signed char*)b, (int*)c, M, N, 384);
    }, M, N, 384, 384, N, N, true, 1.0, false);
    run("bf16 C += A*B", [&](void* a, void* b, void* c) {
        gemm_accumulate<<<dim3(M / 16 / 4, N / 16), 128>>>((const __nv_bfloat16*)a, (const __nv_bfloat16*)b, (float*)c, M, N, K);
    }, M, N, K, K, N, N, false, 1.0, true);
    run("bf16 2x2 tiles per warp, padded", [&](void* a, void* b, void* c) {
        gemm_2x2<<<dim3(M / 32 / 4, N / 32), 128>>>((const __nv_bfloat16*)a, (const __nv_bfloat16*)b, (float*)c, K, K + 32,
                                                    N + 64, N + 16);
    }, M, N, K, K + 32, N + 64, N + 16, false, 1.0, false);
    run("bf16 scaled (not a GEMM)", [&](void* a, void* b, void* c) {
        gemm_scaled<<<dim3(M / 16 / 4, N / 16), 128>>>((const __nv_bfloat16*)a, (const __nv_bfloat16*)b, (float*)c, M, N, K);
    }, M, N, K, K, N, N, false, 2.0, false);
    return check.finish();
}
