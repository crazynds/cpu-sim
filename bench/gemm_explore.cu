// Ablation of the fp32 GEMM written for cucc (bench/gemm_fp32.cu): starts from
// the first version of the duel and applies the changes one at a time, so each
// one's effect can be measured on its own. Prints where the time of one
// multiplication goes (copies, packing, kernel) and the best of a few kernels.
//
//   build/cucc -O3 -std=c++20 bench/gemm_explore.cu -o gemm_explore
//   CPUSIM_BACKEND=inprocess ./gemm_explore <step> [n]      (n: multiple of 1536)
//   ./gemm_explore list
//
// Steps (each includes the ones before it, except where noted):
//   1-copies      12 x 32 per warp, KC 256, packing one thread per element
//                 (64-bit division), cudaMalloc + cudaMemcpy: the first duel
//   2-managed     cudaMallocManaged instead of copies
//   3-warp-pack   packing by warps: one block per panel, contiguous rows
//   4-stcs        packing with __stcs (non-temporal stores; pays only when the
//                 two packed copies, 8 n^2 bytes, do not fit in the last-level cache)
//   5-kc512       KC 512
//   6-6x64        micro-kernel of 6 rows x 64 threads (KC 512)
//   7-kc768       KC 768: the final kernel
//   8-mc24        rows per block 24 instead of 96 (more blocks; pays when the grid
//                 of 96-row blocks is too small for the number of workers)
// Alternatives that did not pay, for comparison with 7:
//   x-12x32-kc768 12 x 32 with KC 768
//   x-kc256       6 x 64 with KC 256
//   x-nc256       6 x 64, KC 768, 256 columns per block
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(call)                                                                            \
    do {                                                                                       \
        cudaError_t e_ = (call);                                                               \
        if (e_ != cudaSuccess) {                                                               \
            std::printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

/* ------------------------------------------------------------------ packing */

// Ap[(m / MR) * K * MR + k * MR + m % MR]: MR rows of A, k-major.
// Bp[(n / NR) * K * NR + k * NR + n % NR]: NR columns of B, k-major.

// One thread per element, with 64-bit division and modulo (step 1).
template <int MR>
__global__ void pack_a_thread(const float* A, float* Ap, int M, int K) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)M * K) return;
    const int m = int(i / K), k = int(i % K);
    Ap[(long)(m / MR) * K * MR + (long)k * MR + m % MR] = A[i];
}
template <int NR>
__global__ void pack_b_thread(const float* B, float* Bp, int K, int N) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)K * N) return;
    const int k = int(i / N), n = int(i % N);
    Bp[(long)(n / NR) * K * NR + (long)k * NR + n % NR] = B[i];
}
// One warp per (micro-panel of MR rows, 32 values of k). Lane l of step j
// writes element 32 j + l of the 32 MR contiguous floats.
template <int MR, bool STREAM>
__global__ void __launch_bounds__(32) pack_a_warp(const float* A, float* Ap, int K) {
    const int lane = threadIdx.x, panel = blockIdx.x, k0 = blockIdx.y * 32;
    float* dst = Ap + (long)panel * K * MR + (long)k0 * MR;
#pragma unroll
    for (int j = 0; j < MR; ++j) {
        const int o = 32 * j + lane;
        const float v = A[(long)(panel * MR + o % MR) * K + k0 + o / MR];
        if (STREAM) __stcs(dst + o, v);
        else dst[o] = v;
    }
}
// One block of NR threads per (panel of NR columns, 256 values of k).
template <int NR, bool STREAM>
__global__ void __launch_bounds__(NR) pack_b_warp(const float* B, float* Bp, int K, int N) {
    const int lane = threadIdx.x, panel = blockIdx.x, k0 = blockIdx.y * 256;
    const float* src = B + (long)k0 * N + panel * NR + lane;
    float* dst = Bp + (long)panel * K * NR + (long)k0 * NR + lane;
    for (int k = 0; k < 256; ++k) {
        if (STREAM) __stcs(dst + k * NR, src[(long)k * N]);
        else dst[k * NR] = src[(long)k * N];
    }
}

/* ------------------------------------------------------------------ kernel */

// Block of NR threads, one column of C each, MR rows per micro-tile; the block
// computes C[m0 : m0 + mc, n0 : n0 + NC] in passes of KC over k.
template <int MR, int NR, int KC, int NC>
__global__ void __launch_bounds__(NR) sgemm(const float* Ap, const float* Bp, float* C, int N, int K, int mc) {
    const int lane = threadIdx.x;
    const int m0 = blockIdx.x * mc, n0 = blockIdx.y * NC;
    for (int pc = 0; pc < K; pc += KC)
        for (int jr = 0; jr < NC; jr += NR) {
            const float* b = Bp + (long)((n0 + jr) / NR) * K * NR + (long)pc * NR + lane;
            for (int ir = 0; ir < mc; ir += MR) {
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

/* ------------------------------------------------------------------ driver */

enum Pack { kThread, kWarp, kStream };

template <int MR, int NR, int KC, int NC>
static void run(const char* step, int n, bool managed, Pack pack, int mc) {
    const size_t elems = size_t(n) * n, bytes = elems * sizeof(float);
    std::vector<float> hA(elems), hB(elems), hC(elems);
    uint32_t s = 12345;
    for (auto& v : hA) s = s * 1664525u + 1013904223u, v = float(int(s >> 28) % 9 - 4);
    for (auto& v : hB) s = s * 1664525u + 1013904223u, v = float(int(s >> 28) % 9 - 4);
    float *A, *B, *C, *Ap, *Bp;
    if (managed) {
        CHECK(cudaMallocManaged(&A, bytes));
        CHECK(cudaMallocManaged(&B, bytes));
        CHECK(cudaMallocManaged(&C, bytes));
        std::copy(hA.begin(), hA.end(), A);  // the application's data, produced in managed memory
        std::copy(hB.begin(), hB.end(), B);
    } else {
        CHECK(cudaMalloc(&A, bytes));
        CHECK(cudaMalloc(&B, bytes));
        CHECK(cudaMalloc(&C, bytes));
    }
    CHECK(cudaMalloc(&Ap, bytes));
    CHECK(cudaMalloc(&Bp, bytes));
    auto upload = [&] {
        if (managed) return;
        CHECK(cudaMemcpy(A, hA.data(), bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(B, hB.data(), bytes, cudaMemcpyHostToDevice));
    };
    auto prepare = [&] {
        if (pack == kThread) {
            pack_a_thread<MR><<<unsigned((elems + 255) / 256), 256>>>(A, Ap, n, n);
            pack_b_thread<NR><<<unsigned((elems + 255) / 256), 256>>>(B, Bp, n, n);
        } else if (pack == kWarp) {
            pack_a_warp<MR, false><<<dim3(n / MR, n / 32), 32>>>(A, Ap, n);
            pack_b_warp<NR, false><<<dim3(n / NR, n / 256), NR>>>(B, Bp, n, n);
        } else {
            pack_a_warp<MR, true><<<dim3(n / MR, n / 32), 32>>>(A, Ap, n);
            pack_b_warp<NR, true><<<dim3(n / NR, n / 256), NR>>>(B, Bp, n, n);
        }
    };
    auto kernel = [&] { sgemm<MR, NR, KC, NC><<<dim3(n / mc, n / NC), NR>>>(Ap, Bp, C, n, n, mc); };
    auto download = [&] {
        if (!managed) CHECK(cudaMemcpy(hC.data(), C, bytes, cudaMemcpyDeviceToHost));
    };

    upload();  // warm-up: JIT and choice of variant
    prepare();
    kernel();
    download();
    CHECK(cudaDeviceSynchronize());

    const double t0 = now_s();
    upload();
    const double t1 = now_s();
    prepare();
    CHECK(cudaDeviceSynchronize());
    const double t2 = now_s();
    kernel();
    CHECK(cudaDeviceSynchronize());
    const double t3 = now_s();
    download();
    const double t4 = now_s();

    double best = 1e30;
    for (int r = 0; r < 3; ++r) {
        const double a = now_s();
        kernel();
        CHECK(cudaDeviceSynchronize());
        best = std::min(best, now_s() - a);
    }
    if (managed) std::copy(C, C + elems, hC.begin());
    int bad = 0;  // 16 rows, including the first and the last, checked exactly
    for (int t = 0; t < 16; ++t) {
        const int i = t == 0 ? 0 : t == 1 ? n - 1 : int((size_t(t) * 2654435761u) % unsigned(n));
        std::vector<double> row(n, 0.0);
        for (int k = 0; k < n; ++k) {
            const double a = hA[size_t(i) * n + k];
            for (int j = 0; j < n; ++j) row[j] += a * hB[size_t(k) * n + j];
        }
        for (int j = 0; j < n; ++j) bad += hC[size_t(i) * n + j] != row[j];
    }
    std::printf("%-14s n=%5d  copies %7.2f  pack %6.2f  kernel %8.2f  total %8.2f ms | best kernel %8.2f ms %7.1f GFLOPS  %s\n",
                step, n, (t1 - t0 + t4 - t3) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t0) * 1e3, best * 1e3,
                2.0 * n * n * n / best / 1e9, bad ? "FAIL" : "ok");
    cudaFree(A), cudaFree(B), cudaFree(C), cudaFree(Ap), cudaFree(Bp);
}

int main(int argc, char** argv) {
    const std::string step = argc > 1 ? argv[1] : "list";
    const int n = argc > 2 ? std::atoi(argv[2]) : 3072;
    if (n % 1536 != 0) {
        std::printf("n must be a multiple of 1536\n");
        return 1;
    }
    const char* s = step.c_str();
    if (step == "1-copies") run<12, 32, 256, 512>(s, n, false, kThread, 96);
    else if (step == "2-managed") run<12, 32, 256, 512>(s, n, true, kThread, 96);
    else if (step == "3-warp-pack") run<12, 32, 256, 512>(s, n, true, kWarp, 96);
    else if (step == "4-stcs") run<12, 32, 256, 512>(s, n, true, kStream, 96);
    else if (step == "5-kc512") run<12, 32, 512, 512>(s, n, true, kStream, 96);
    else if (step == "6-6x64") run<6, 64, 512, 512>(s, n, true, kStream, 96);
    else if (step == "7-kc768") run<6, 64, 768, 512>(s, n, true, kStream, 96);
    else if (step == "8-mc24") run<6, 64, 768, 512>(s, n, true, kStream, 24);
    else if (step == "x-12x32-kc768") run<12, 32, 768, 512>(s, n, true, kStream, 96);
    else if (step == "x-kc256") run<6, 64, 256, 512>(s, n, true, kStream, 96);
    else if (step == "x-nc256") run<6, 64, 768, 256>(s, n, true, kStream, 96);
    else {
        std::printf("steps: 1-copies 2-managed 3-warp-pack 4-stcs 5-kc512 6-6x64 7-kc768 8-mc24\n"
                    "alternatives: x-12x32-kc768 x-kc256 x-nc256\n");
        return step == "list" ? 0 : 1;
    }
}
