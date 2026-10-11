// GEMM written for the CPU simulator's execution model rather than for a GPU:
// fp32 on the SIMT path and bf16 on the tensor core path, reporting GFLOPS.
// It is valid CUDA and runs on a GPU too, just not tuned for one.
//
// How the code maps onto the simulator (cucc):
//   - A block is one warp (32 threads), without barriers or shared memory:
//     the 32-lane variant runs it from start to end, no context switches.
//   - fp32: each thread is one column of C and accumulates 12 rows. With 32
//     lanes an accumulator is 2 zmm registers, so 24 zmm of accumulators, 2
//     for B and 1 for a broadcast: no spills. Per step of k, A's value is the
//     same for the whole warp (one scalar load and broadcast) and B's row is
//     contiguous across lanes (two zmm loads): 24 FMAs per 2 vector loads.
//   - bf16: each warp computes one 16 x 16 tile of C with one accumulator
//     fragment. With 32 lanes it is 16 zmm registers, so it stays in
//     registers through the whole loop over k (more accumulators would
//     spill); each mma_sync becomes 128 vdpbf16ps inline (include/mma.h).
//     A is loaded one step of k ahead: the mma broadcasts A's pairs from its
//     fragment in memory, and by then the stores that filled it are done.
//   - Caches: A and B are first packed (by kernels) so that what a thread
//     reads at each step is contiguous: fp32 micro-panels of 12 rows x k and
//     k x 32 columns (as BLIS does), bf16 16 x 16 tiles in fragment order.
//     The fp32 kernel goes through k in passes of KC, so that the 32 KiB
//     micro-panel of B is reused from L1 by the 8 micro-panels of A of a
//     block, the 96 KiB block of A from L2, and blocks next to each other
//     share their panel of B in L3.
//
//   cpu_gemm [fp32|bf16|all] [iterations]
#include <cuda_bf16.h>
#include <mma.h>

#include <string>

#include "common.cuh"

using namespace nvcuda;

/* ------------------------------------------------------------------ fp32 */

constexpr int MR = 12, NR = 32;  // micro-tile: 12 rows x 32 columns, one column per thread
constexpr int KC = 256;          // k per pass
constexpr int MC = 96;           // rows per block (8 micro-panels of A)
constexpr int NC = 512;          // columns per block (16 micro-panels of B)

// Ap[(m / 12) * K * 12 + k * 12 + m % 12]: 12 rows of A, k-major.
__global__ void pack_a32(const float* A, float* Ap, int M, int K) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)M * K) return;
    const int m = int(i / K), k = int(i % K);
    Ap[(long)(m / MR) * K * MR + (long)k * MR + m % MR] = A[i];
}
// Bp[(n / 32) * K * 32 + k * 32 + n % 32]: 32 columns of B, k-major.
__global__ void pack_b32(const float* B, float* Bp, int K, int N) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)K * N) return;
    const int k = int(i / N), n = int(i % N);
    Bp[(long)(n / NR) * K * NR + (long)k * NR + n % NR] = B[i];
}

// One warp per block: C[m0 : m0 + MC, n0 : n0 + NC] = A * B.
__global__ void __launch_bounds__(32) sgemm(const float* Ap, const float* Bp, float* C, int N, int K) {
    const int lane = threadIdx.x;
    const int m0 = blockIdx.x * MC, n0 = blockIdx.y * NC;
    for (int pc = 0; pc < K; pc += KC)
        for (int jr = 0; jr < NC; jr += NR) {
            const float* b = Bp + (long)((n0 + jr) / NR) * K * NR + (long)pc * NR + lane;
            for (int ir = 0; ir < MC; ir += MR) {
                const float* a = Ap + (long)((m0 + ir) / MR) * K * MR + (long)pc * MR;
                float* c = C + (long)(m0 + ir) * N + n0 + jr + lane;
                float acc[MR];
#pragma unroll
                for (int r = 0; r < MR; ++r) acc[r] = pc == 0 ? 0.0f : c[(long)r * N];
#pragma unroll 4
                for (int k = 0; k < KC; ++k) {
                    const float bv = b[k * NR];
#pragma unroll
                    for (int r = 0; r < MR; ++r) acc[r] = fmaf(a[k * MR + r], bv, acc[r]);
                }
#pragma unroll
                for (int r = 0; r < MR; ++r) c[(long)r * N] = acc[r];
            }
        }
}

/* ------------------------------------------------------------------ bf16 */


// 16 x 16 tiles, each contiguous and row-major (A: m x k, B: k x n).
__global__ void pack_a16(const __nv_bfloat16* A, __nv_bfloat16* Ap, int M, int K) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)M * K) return;
    const int m = int(i / K), k = int(i % K);
    Ap[((long)(m / 16) * (K / 16) + k / 16) * 256 + (m % 16) * 16 + k % 16] = A[i];
}
__global__ void pack_b16(const __nv_bfloat16* B, __nv_bfloat16* Bp, int K, int N) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)K * N) return;
    const int k = int(i / N), n = int(i % N);
    Bp[((long)(n / 16) * (K / 16) + k / 16) * 256 + (k % 16) * 16 + n % 16] = B[i];
}

__global__ void __launch_bounds__(32) bgemm(const __nv_bfloat16* Ap, const __nv_bfloat16* Bp, float* C, int N, int K) {
    const int tn = blockIdx.x, tm = blockIdx.y;  // consecutive blocks share the row panel of A
    const int KT = K / 16;
    const __nv_bfloat16* ap = Ap + (long)tm * KT * 256;
    const __nv_bfloat16* bp = Bp + (long)tn * KT * 256;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a0, a1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
    wmma::load_matrix_sync(a0, ap, 16);
    for (int kt = 0; kt < KT; kt += 2) {  // K is a multiple of 32
        wmma::load_matrix_sync(a1, ap + (kt + 1) * 256, 16);
        wmma::load_matrix_sync(b, bp + kt * 256, 16);
        wmma::mma_sync(acc, a0, b, acc);
        if (kt + 2 < KT) wmma::load_matrix_sync(a0, ap + (kt + 2) * 256, 16);
        wmma::load_matrix_sync(b, bp + (kt + 1) * 256, 16);
        wmma::mma_sync(acc, a1, b, acc);
    }
    wmma::store_matrix_sync(C + (long)tm * 16 * N + tn * 16, acc, N, wmma::mem_row_major);
}

/* ------------------------------------------------------------------ host */

// Small integers: every sum is exact in fp32, so C must equal the reference.
static std::vector<int> small_ints(size_t n, uint64_t seed) {
    Lcg rng(seed);
    std::vector<int> v(n);
    for (auto& x : v) x = int(rng.next_u32() % 9) - 4;
    return v;
}

// Times `launch` (best of `iterations`, after a warm-up that also lets the
// simulator compile and choose its variant) and checks 64 rows of C.
template <typename F>
static bool measure(const char* name, int M, int N, int K, int iterations, const std::vector<int>& a,
                    const std::vector<int>& b, const DeviceBuffer<float>& dC, F&& launch) {
    launch();
    CUDA_CHECK_KERNEL();
    double best = 1e30;
    for (int it = 0; it < iterations; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        launch();
        CUDA_CHECK_KERNEL();
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::vector<float> c;
    dC.download(c);
    size_t bad = 0;
    for (int t = 0; t < 64; ++t) {
        const int i = int((size_t(t) * 2654435761u) % unsigned(M));
        std::vector<long long> row(N, 0);
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j) row[j] += (long long)a[size_t(i) * K + k] * b[size_t(k) * N + j];
        for (int j = 0; j < N; ++j) bad += (double)c[size_t(i) * N + j] != (double)row[j];
    }
    std::printf("  %-6s %5d x %5d x %5d  %9.3f ms  %8.1f GFLOPS  %s\n", name, M, N, K, best * 1e3,
                2.0 * M * N * K / best / 1e9, bad ? "FAIL" : "ok");
    return bad == 0;
}

static bool run_fp32(int iterations) {
    const int M = SAMPLE_SIZE(4032, 192), N = SAMPLE_SIZE(4096, 512), K = SAMPLE_SIZE(4096, 512);
    const auto a = small_ints(size_t(M) * K, 1), b = small_ints(size_t(K) * N, 2);
    std::vector<float> fa(a.begin(), a.end()), fb(b.begin(), b.end());
    DeviceBuffer<float> dA(fa.size()), dB(fb.size()), dAp(fa.size()), dBp(fb.size()), dC(size_t(M) * N);
    dA.upload(fa);
    dB.upload(fb);
    pack_a32<<<unsigned((fa.size() + 255) / 256), 256>>>(dA.get(), dAp.get(), M, K);
    pack_b32<<<unsigned((fb.size() + 255) / 256), 256>>>(dB.get(), dBp.get(), K, N);
    CUDA_CHECK_KERNEL();
    return measure("fp32", M, N, K, iterations, a, b, dC,
                   [&] { sgemm<<<dim3(M / MC, N / NC), 32>>>(dAp.get(), dBp.get(), dC.get(), N, K); });
}

static bool run_bf16(int cc, int iterations) {
    if (cc < 80) {
        std::printf("  bf16   skipped: tensor cores for bf16 need compute capability 8.0\n");
        return true;
    }
    const int M = SAMPLE_SIZE(4096, 256), N = SAMPLE_SIZE(4096, 256), K = SAMPLE_SIZE(4096, 256);
    const auto a = small_ints(size_t(M) * K, 3), b = small_ints(size_t(K) * N, 4);
    std::vector<__nv_bfloat16> ha(a.size()), hb(b.size());
    for (size_t i = 0; i < a.size(); ++i) ha[i] = __float2bfloat16(float(a[i]));
    for (size_t i = 0; i < b.size(); ++i) hb[i] = __float2bfloat16(float(b[i]));
    DeviceBuffer<__nv_bfloat16> dA(ha.size()), dB(hb.size()), dAp(ha.size()), dBp(hb.size());
    DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(ha);
    dB.upload(hb);
    pack_a16<<<unsigned((ha.size() + 255) / 256), 256>>>(dA.get(), dAp.get(), M, K);
    pack_b16<<<unsigned((hb.size() + 255) / 256), 256>>>(dB.get(), dBp.get(), K, N);
    CUDA_CHECK_KERNEL();
    return measure("bf16", M, N, K, iterations, a, b, dC,
                   [&] { bgemm<<<dim3(N / 16, M / 16), 32>>>(dAp.get(), dBp.get(), dC.get(), N, K); });
}

int main(int argc, char** argv) {
    const std::string which = argc > 1 ? argv[1] : "all";
    const int iterations = argc > 2 ? std::atoi(argv[2]) : SAMPLE_SIZE(10, 2);
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s, compute capability %d.%d\n", prop.name, prop.major, prop.minor);
    bool ok = true;
    if (which == "all" || which == "fp32") ok &= run_fp32(iterations);
    if (which == "all" || which == "bf16") ok &= run_bf16(prop.major * 10 + prop.minor, iterations);
    return finish("cpu_gemm", ok);
}
