// fp32 GEMM for warps of 16 threads (build with CPUSIM_WARP_SIZE=16): a block
// is one warp of 16 threads, one zmm register per value. Each thread owns NJ
// columns of the micro-tile (lane, lane + 16, ...) and MR rows, so the
// micro-tile is MR x 16*NJ with MR*NJ accumulators, one zmm each. Per step of
// k: NJ contiguous loads of B, MR broadcasts of A, MR*NJ FMAs.
// Sizes and blocking are macros, to sweep them:
//   cucc -DMR=6 -DNJ=4 -DKC=256 -DMC=96 -DNC=512 bench/gemm16.cu
//   gemm16 [iterations] [M N K]
#include "../samples/common/common.cuh"

#ifndef MR
#define MR 6
#endif
#ifndef NJ
#define NJ 4
#endif
#ifndef KC
#define KC 256
#endif
#ifndef MC
#define MC 96
#endif
#ifndef NC
#define NC 512
#endif
#ifndef WARP
#define WARP 16  // threads per block; the kernel is written for warps of 16
#endif

constexpr int NR = WARP * NJ;  // micro-tile columns
static_assert(MC % MR == 0 && NC % NR == 0, "blocks are whole micro-tiles");

// Ap[(m / MR) * K * MR + k * MR + m % MR]: MR rows of A, k-major.
__global__ void pack_a(const float* A, float* Ap, int M, int K) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)M * K) return;
    const int m = int(i / K), k = int(i % K);
    Ap[(long)(m / MR) * K * MR + (long)k * MR + m % MR] = A[i];
}
// Bp[(n / NR) * K * NR + k * NR + n % NR]: NR columns of B, k-major.
__global__ void pack_b(const float* B, float* Bp, int K, int N) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)K * N) return;
    const int k = int(i / N), n = int(i % N);
    Bp[(long)(n / NR) * K * NR + (long)k * NR + n % NR] = B[i];
}

// One warp per block: C[m0 : m0 + MC, n0 : n0 + NC] = A * B.
__global__ void __launch_bounds__(WARP) sgemm(const float* Ap, const float* Bp, float* C, int N, int K) {
    const int lane = threadIdx.x;
    const int m0 = blockIdx.x * MC, n0 = blockIdx.y * NC;
    for (int pc = 0; pc < K; pc += KC)
        for (int jr = 0; jr < NC; jr += NR) {
            const float* b = Bp + (long)((n0 + jr) / NR) * K * NR + (long)pc * NR + lane;
            for (int ir = 0; ir < MC; ir += MR) {
                const float* a = Ap + (long)((m0 + ir) / MR) * K * MR + (long)pc * MR;
                float* c = C + (long)(m0 + ir) * N + n0 + jr + lane;
                float acc[MR][NJ];
#pragma unroll
                for (int r = 0; r < MR; ++r)
#pragma unroll
                    for (int j = 0; j < NJ; ++j) acc[r][j] = pc == 0 ? 0.0f : c[(long)r * N + j * WARP];
#pragma unroll 2
                for (int k = 0; k < KC; ++k) {
                    float bv[NJ];
#pragma unroll
                    for (int j = 0; j < NJ; ++j) bv[j] = b[k * NR + j * WARP];
#pragma unroll
                    for (int r = 0; r < MR; ++r) {
                        const float av = a[k * MR + r];
#pragma unroll
                        for (int j = 0; j < NJ; ++j) acc[r][j] = fmaf(av, bv[j], acc[r][j]);
                    }
                }
#pragma unroll
                for (int r = 0; r < MR; ++r)
#pragma unroll
                    for (int j = 0; j < NJ; ++j) c[(long)r * N + j * WARP] = acc[r][j];
            }
        }
}

static std::vector<int> small_ints(size_t n, uint64_t seed) {
    Lcg rng(seed);
    std::vector<int> v(n);
    for (auto& x : v) x = int(rng.next_u32() % 9) - 4;
    return v;
}

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 5;
    // gemm16 [iterations] [M N K]: rounded down to whole blocks.
    const int m = argc > 4 ? std::atoi(argv[2]) : 4032, nn = argc > 4 ? std::atoi(argv[3]) : 4096,
              kk = argc > 4 ? std::atoi(argv[4]) : 4096;
    const int M = MC * (m / MC), N = NC * (nn / NC), K = KC * (kk / KC);
    const auto a = small_ints(size_t(M) * K, 1), b = small_ints(size_t(K) * N, 2);
    std::vector<float> fa(a.begin(), a.end()), fb(b.begin(), b.end());
    DeviceBuffer<float> dA(fa.size()), dB(fb.size()), dAp(fa.size()), dBp(fb.size()), dC(size_t(M) * N);
    dA.upload(fa);
    dB.upload(fb);
    pack_a<<<unsigned((fa.size() + 255) / 256), 256>>>(dA.get(), dAp.get(), M, K);
    pack_b<<<unsigned((fb.size() + 255) / 256), 256>>>(dB.get(), dBp.get(), K, N);
    CUDA_CHECK_KERNEL();
    auto launch = [&] { sgemm<<<dim3(M / MC, N / NC), WARP>>>(dAp.get(), dBp.get(), dC.get(), N, K); };
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
    for (int t = 0; t < 32; ++t) {
        const int i = int((size_t(t) * 2654435761u) % unsigned(M));
        std::vector<long long> row(N, 0);
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j) row[j] += (long long)a[size_t(i) * K + k] * b[size_t(k) * N + j];
        for (int j = 0; j < N; ++j) bad += (double)c[size_t(i) * N + j] != (double)row[j];
    }
    std::printf("GEMM MR=%d NJ=%d (%dx%d) KC=%d MC=%d NC=%d warp=%d  %dx%dx%d  %.3f ms  %.1f GFLOPS  %s\n", MR, NJ, MR,
                NR, KC, MC, NC, WARP, M, N, K, best * 1e3, 2.0 * M * N * K / best / 1e9, bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
