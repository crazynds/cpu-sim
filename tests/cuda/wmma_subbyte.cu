// Tensor cores, experimental sub-byte types (nvcuda::wmma::experimental):
// s4 and u4 -> int (8x8x32, mma_sync) and b1 -> int (8x8x128, bmma_sync with
// xor and with and, popcount). A is row-major, B column-major, packed (8
// values of 4 bits or 32 bits to a word), as CUDA requires. One warp per 8x8
// tile of D = A * B + C; the result must match the host exactly, or, below
// compute capability 7.5, the launch must fail with
// cudaErrorNoKernelImageForDevice.
#include <mma.h>

#include <cstdint>
#include <vector>

#include "check.cuh"

using namespace nvcuda;
namespace xp = nvcuda::wmma::experimental;

constexpr int M = 32, N = 32;

// K: elements per row of A (and column of B); bits per element: 4 or 1.
template <typename T, int TK, int BITS>
__global__ void gemm(const unsigned* a, const unsigned* b, const int* c, int* d, int K, int op) {
    const int m0 = blockIdx.x * 8, n0 = blockIdx.y * 8;
    wmma::fragment<wmma::matrix_a, 8, 8, TK, T, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 8, 8, TK, T, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 8, 8, TK, int> acc;
    wmma::load_matrix_sync(acc, c + m0 * N + n0, N, wmma::mem_row_major);
    for (int k0 = 0; k0 < K; k0 += TK) {
        wmma::load_matrix_sync(fa, a + (m0 * K + k0) * BITS / 32, K);
        wmma::load_matrix_sync(fb, b + (n0 * K + k0) * BITS / 32, K);
        if constexpr (BITS == 1)
            wmma::bmma_sync(acc, fa, fb, acc, op ? xp::bmmaBitOpAND : xp::bmmaBitOpXOR, xp::bmmaAccumulateOpPOPC);
        else
            wmma::mma_sync(acc, fa, fb, acc);
    }
    wmma::store_matrix_sync(d + m0 * N + n0, acc, N, wmma::mem_row_major);
}

// kind: 0 s4, 1 u4, 2 b1 xor, 3 b1 and.
template <typename T, int TK, int BITS>
static void run(Checker& check, const char* name, int kind, int cc) {
    const int K = BITS == 1 ? 512 : 256;
    std::vector<int> av(M * K), bv(K * N), cv(M * N);  // logical values; b: bv[n * K + k]
    std::vector<unsigned> a(M * K * BITS / 32), b(N * K * BITS / 32);
    uint32_t s = 777u + kind;
    auto next = [&] { return (s = s * 1664525u + 1013904223u) >> 8; };
    const int lo = kind == 0 ? -8 : 0, hi = kind == 0 ? 7 : BITS == 1 ? 1 : 15;
    auto pack = [&](std::vector<unsigned>& w, int i, int v) {  // element i of a packed row-major array
        w[i * BITS / 32] |= (unsigned(v) & ((1u << BITS) - 1)) << (i * BITS % 32);
    };
    for (int i = 0; i < M * K; ++i) av[i] = lo + int(next() % unsigned(hi - lo + 1)), pack(a, i, av[i]);
    for (int i = 0; i < N * K; ++i) bv[i] = lo + int(next() % unsigned(hi - lo + 1)), pack(b, i, bv[i]);
    for (int& v : cv) v = int(next() % 41) - 20;
    unsigned *d_a, *d_b;
    int *d_c, *d_d;
    CHECK_CUDA(cudaMalloc(&d_a, a.size() * 4));
    CHECK_CUDA(cudaMalloc(&d_b, b.size() * 4));
    CHECK_CUDA(cudaMalloc(&d_c, cv.size() * 4));
    CHECK_CUDA(cudaMalloc(&d_d, cv.size() * 4));
    CHECK_CUDA(cudaMemcpy(d_a, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, cv.data(), cv.size() * 4, cudaMemcpyHostToDevice));
    gemm<T, TK, BITS><<<dim3(M / 8, N / 8), 32>>>(d_a, d_b, d_c, d_d, K, kind == 3);
    const cudaError_t launch = cudaGetLastError();
    char what[96];
    std::snprintf(what, sizeof what, "%s: launch on cc %d (needs 75)", name, cc);
    if (cc < 75) {
        check.expect(launch == cudaErrorNoKernelImageForDevice, what, launch, launch, cudaErrorNoKernelImageForDevice);
    } else {
        check.expect(launch == cudaSuccess, what, launch, launch, cudaSuccess);
        std::vector<int> d(cv.size());
        CHECK_CUDA(cudaMemcpy(d.data(), d_d, d.size() * 4, cudaMemcpyDeviceToHost));
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                long want = cv[m * N + n];
                for (int k = 0; k < K; ++k) {
                    const int x = av[m * K + k], y = bv[n * K + k];
                    want += kind == 2 ? (x ^ y) : kind == 3 ? (x & y) : x * y;
                }
                check.expect(d[m * N + n] == want, name, m * N + n, d[m * N + n], double(want));
            }
    }
    cudaFree(d_a), cudaFree(d_b), cudaFree(d_c), cudaFree(d_d);
}

int main() {
    Checker check("wmma_subbyte");
    cudaDeviceProp p;
    CHECK_CUDA(cudaGetDeviceProperties(&p, 0));
    const int cc = p.major * 10 + p.minor;
    std::printf("compute capability %d.%d\n", p.major, p.minor);
    run<xp::precision::s4, 32, 4>(check, "s4->int", 0, cc);
    run<xp::precision::u4, 32, 4>(check, "u4->int", 1, cc);
    run<xp::precision::b1, 128, 1>(check, "b1 xor popc", 2, cc);
    run<xp::precision::b1, 128, 1>(check, "b1 and popc", 3, cc);
    return check.finish();
}
