// fp32 matrix multiplication, C = A * B (n x n), three contenders:
//   cuda: the best each CUDA toolchain offers
//         - nvcc (GPU): cuBLAS with TF32 tensor cores (cublasGemmEx,
//           CUBLAS_COMPUTE_32F_FAST_TF32);
//         - cucc (CPU simulator): the kernel written for the simulator's
//           execution model (samples/cpu_gemm.cu): packing kernels plus a
//           one-warp-per-block BLIS-like micro-kernel, 12 x 32 per warp;
//   omp:  C++ with OpenMP and AVX-512, BLIS-like (packing, KC/MC/NC cache
//         blocks, 8 x 32 micro-kernel in 16 zmm registers).
// Inputs are small integers, so every product and sum is exact in fp32 (and
// in TF32, whose 10-bit mantissa holds them): every result must equal the
// reference exactly.
//
// Per size: time per kernel (mean of the timed repetitions), total time of
// one multiplication from host data to host result (copies and packing
// included), and GFLOPS of the kernel.
//
//   build/cucc -O3 -std=c++20 -fopenmp bench/gemm_fp32.cu -o gemm_cpu
//   nvcc -O3 -std=c++20 -arch=native -Xcompiler -fopenmp bench/gemm_fp32.cu -lcublas -o gemm_gpu
//   ./gemm_cpu cuda|omp [sizes...]
#include <omp.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#ifdef __NVCC__
#include <cublas_v2.h>
#endif

#define CHECK(call)                                                                            \
    do {                                                                                       \
        cudaError_t e_ = (call);                                                               \
        if (e_ != cudaSuccess) {                                                               \
            std::printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

/* ------------------------------------------------- cucc: written for the simulator */

constexpr int SMR = 12, SNR = 32;            // micro-tile: 12 rows x 32 columns, one column per thread
constexpr int SKC = 256, SMC = 96, SNC = 512;  // k per pass, rows and columns per block

__global__ void pack_a(const float* A, float* Ap, int M, int K) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)M * K) return;
    const int m = int(i / K), k = int(i % K);
    Ap[(long)(m / SMR) * K * SMR + (long)k * SMR + m % SMR] = A[i];
}
__global__ void pack_b(const float* B, float* Bp, int K, int N) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)K * N) return;
    const int k = int(i / N), n = int(i % N);
    Bp[(long)(n / SNR) * K * SNR + (long)k * SNR + n % SNR] = B[i];
}
__global__ void __launch_bounds__(32) sgemm(const float* Ap, const float* Bp, float* C, int N, int K) {
    const int lane = threadIdx.x;
    const int m0 = blockIdx.x * SMC, n0 = blockIdx.y * SNC;
    for (int pc = 0; pc < K; pc += SKC)
        for (int jr = 0; jr < SNC; jr += SNR) {
            const float* b = Bp + (long)((n0 + jr) / SNR) * K * SNR + (long)pc * SNR + lane;
            for (int ir = 0; ir < SMC; ir += SMR) {
                const float* a = Ap + (long)((m0 + ir) / SMR) * K * SMR + (long)pc * SMR;
                float* c = C + (long)(m0 + ir) * N + n0 + jr + lane;
                float acc[SMR];
#pragma unroll
                for (int r = 0; r < SMR; ++r) acc[r] = pc == 0 ? 0.0f : c[(long)r * N];
#pragma unroll 4
                for (int k = 0; k < SKC; ++k) {
                    const float bv = b[k * SNR];
#pragma unroll
                    for (int r = 0; r < SMR; ++r) acc[r] = fmaf(a[k * SMR + r], bv, acc[r]);
                }
#pragma unroll
                for (int r = 0; r < SMR; ++r) c[(long)r * N] = acc[r];
            }
        }
}

/* ------------------------------------------------------- OpenMP + AVX-512 */

constexpr int MR = 8, NR = 32;               // register block
constexpr int KC = 256, MC = 128, NC = 512;  // cache blocks
typedef float v16 __attribute__((vector_size(64)));
typedef float v16u __attribute__((vector_size(64), aligned(4)));

__attribute__((target("avx512f,fma"))) static void micro_kernel(int kc, const float* __restrict a,
                                                                 const float* __restrict b, float* c, int ldc,
                                                                 bool accumulate) {
    v16 acc[MR][2] = {};
    for (int p = 0; p < kc; p++) {
        v16 b0 = *(const v16*)(b + p * NR);
        v16 b1 = *(const v16*)(b + p * NR + 16);
        for (int i = 0; i < MR; i++) {
            float ai = a[p * MR + i];
            acc[i][0] += ai * b0;
            acc[i][1] += ai * b1;
        }
    }
    for (int i = 0; i < MR; i++) {
        v16u* row = (v16u*)(c + (size_t)i * ldc);
        if (accumulate) {
            row[0] += acc[i][0];
            row[1] += acc[i][1];
        } else {
            row[0] = acc[i][0];
            row[1] = acc[i][1];
        }
    }
}
static void omp_pack_b(const float* B, float* Bp, int n, int k0, int kc, int panel) {
    int j0 = panel * NR;
    float* dst = Bp + (size_t)panel * kc * NR;
    for (int p = 0; p < kc; p++) {
        const float* src = B + (size_t)(k0 + p) * n + j0;
        for (int j = 0; j < NR; j++) dst[p * NR + j] = j0 + j < n ? src[j] : 0.0f;
    }
}
static void omp_pack_a(const float* A, float* Ap, int n, int i0, int mc, int k0, int kc) {
    for (int ir = 0; ir < mc; ir += MR) {
        float* dst = Ap + (size_t)(ir / MR) * kc * MR;
        for (int p = 0; p < kc; p++)
            for (int i = 0; i < MR; i++) dst[p * MR + i] = ir + i < mc ? A[(size_t)(i0 + ir + i) * n + k0 + p] : 0.0f;
    }
}
static void matmul_omp_fast(const float* A, const float* B, float* C, int n) {
    int b_panels = (n + NR - 1) / NR;
    float* Bp = (float*)std::aligned_alloc(64, (size_t)b_panels * KC * NR * sizeof(float));
    int m_blocks = (n + MC - 1) / MC, n_blocks = (n + NC - 1) / NC;
#pragma omp parallel
    {
        float* Ap = (float*)std::aligned_alloc(64, (size_t)MC * KC * sizeof(float));
        alignas(64) float edge[MR * NR];
        for (int k0 = 0; k0 < n; k0 += KC) {
            int kc = n - k0 < KC ? n - k0 : KC;
            bool accumulate = k0 > 0;
#pragma omp for schedule(static)
            for (int panel = 0; panel < b_panels; panel++) omp_pack_b(B, Bp, n, k0, kc, panel);
#pragma omp for collapse(2) schedule(dynamic)
            for (int mb = 0; mb < m_blocks; mb++)
                for (int nb = 0; nb < n_blocks; nb++) {
                    int i0 = mb * MC, mc = n - i0 < MC ? n - i0 : MC;
                    int j0 = nb * NC, nc = n - j0 < NC ? n - j0 : NC;
                    omp_pack_a(A, Ap, n, i0, mc, k0, kc);
                    for (int jr = 0; jr < nc; jr += NR) {
                        const float* bp = Bp + (size_t)((j0 + jr) / NR) * kc * NR;
                        for (int ir = 0; ir < mc; ir += MR) {
                            const float* ap = Ap + (size_t)(ir / MR) * kc * MR;
                            float* c = C + (size_t)(i0 + ir) * n + j0 + jr;
                            int mr = mc - ir < MR ? mc - ir : MR, nr = nc - jr < NR ? nc - jr : NR;
                            if (mr == MR && nr == NR) {
                                micro_kernel(kc, ap, bp, c, n, accumulate);
                            } else {
                                micro_kernel(kc, ap, bp, edge, NR, false);
                                for (int i = 0; i < mr; i++)
                                    for (int j = 0; j < nr; j++)
                                        c[(size_t)i * n + j] = (accumulate ? c[(size_t)i * n + j] : 0.0f) + edge[i * NR + j];
                            }
                        }
                    }
                }
        }
        std::free(Ap);
    }
    std::free(Bp);
}

/* ------------------------------------------------------------------ driver */

struct Result {
    double kernel_s, total_s;
};

// One multiplication of the CUDA contender, host to host; then `reps`
// kernels timed alone.
static Result run_cuda(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int n, int reps) {
    const size_t bytes = size_t(n) * n * sizeof(float);
    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));
#ifdef __NVCC__
    cublasHandle_t h;
    cublasCreate(&h);
    const float one = 1.0f, zero = 0.0f;
    // Row-major C = A * B is column-major C^T = B^T * A^T.
    auto kernel = [&] {
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, dB, CUDA_R_32F, n, dA, CUDA_R_32F, n, &zero, dC,
                     CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
    };
    auto prepare = [&] {};
#else
    float *dAp, *dBp;
    CHECK(cudaMalloc(&dAp, bytes));
    CHECK(cudaMalloc(&dBp, bytes));
    auto prepare = [&] {
        pack_a<<<unsigned((size_t(n) * n + 255) / 256), 256>>>(dA, dAp, n, n);
        pack_b<<<unsigned((size_t(n) * n + 255) / 256), 256>>>(dB, dBp, n, n);
    };
    auto kernel = [&] { sgemm<<<dim3(n / SMC, n / SNC), 32>>>(dAp, dBp, dC, n, n); };
#endif
    // Warm-up (library setup on the GPU; JIT and choice of variant on the CPU).
    CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
    prepare();
    kernel();
    CHECK(cudaDeviceSynchronize());

    double t0 = now_s();
    CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
    prepare();
    kernel();
    CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));
    const double total = now_s() - t0;

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0));
    for (int r = 0; r < reps; ++r) kernel();
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CHECK(cudaGetLastError());
#ifdef __NVCC__
    cublasDestroy(h);
#else
    cudaFree(dAp), cudaFree(dBp);
#endif
    cudaFree(dA), cudaFree(dB), cudaFree(dC);
    return {ms / 1e3 / reps, total};
}

static Result run_omp(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int n, int reps) {
    matmul_omp_fast(A.data(), B.data(), C.data(), n);  // warm-up
    const double t0 = now_s();
    for (int r = 0; r < reps; ++r) matmul_omp_fast(A.data(), B.data(), C.data(), n);
    const double each = (now_s() - t0) / reps;
    return {each, each};  // host to host already: the call is the whole multiplication
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "cuda";
    std::vector<int> sizes;
    for (int i = 2; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {1536, 3072, 4608, 6144};
    if (mode == "cuda") {
        cudaDeviceProp p;
        CHECK(cudaGetDeviceProperties(&p, 0));
        std::printf("cuda on %s (%s)\n", p.name,
#ifdef __NVCC__
                    "cuBLAS, TF32 tensor cores"
#else
                    "cucc, kernel written for the simulator"
#endif
        );
    } else {
        std::printf("omp, %d threads\n", omp_get_max_threads());
    }
    std::printf("%6s  %12s  %12s  %10s  %s\n", "n", "kernel (ms)", "total (ms)", "GFLOPS", "check");
    for (int n : sizes) {
        if (n % 1536 != 0) {
            std::printf("%6d  skipped: n must be a multiple of 1536\n", n);
            continue;
        }
        std::vector<float> A(size_t(n) * n), B(size_t(n) * n), C(size_t(n) * n);
        uint32_t s = 12345;
        for (auto& v : A) s = s * 1664525u + 1013904223u, v = float(int(s >> 28) % 9 - 4);
        for (auto& v : B) s = s * 1664525u + 1013904223u, v = float(int(s >> 28) % 9 - 4);
        const double flops = 2.0 * n * n * n;
        const int reps = flops < 2e10 ? 20 : 5;
        Result r = mode == "cuda" ? run_cuda(A, B, C, n, reps) : run_omp(A, B, C, n, reps);
        // 32 rows checked exactly.
        int bad = 0;
        for (int t = 0; t < 32; ++t) {
            const int i = int((size_t(t) * 2654435761u) % unsigned(n));
            std::vector<double> row(n, 0.0);
            for (int k = 0; k < n; ++k) {
                const double a = A[size_t(i) * n + k];
                for (int j = 0; j < n; ++j) row[j] += a * B[size_t(k) * n + j];
            }
            for (int j = 0; j < n; ++j) bad += C[size_t(i) * n + j] != row[j];
        }
        std::printf("%6d  %12.3f  %12.3f  %10.1f  %s\n", n, r.kernel_s * 1e3, r.total_s * 1e3, flops / r.kernel_s / 1e9,
                    bad ? "FAIL" : "ok");
    }
}
