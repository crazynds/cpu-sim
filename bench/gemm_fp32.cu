// fp32 matrix multiplication, C = A * B (n x n), three contenders:
//   cuda: the best each CUDA toolchain offers
//         - nvcc (GPU): cuBLAS with TF32 tensor cores (cublasGemmEx,
//           CUBLAS_COMPUTE_32F_FAST_TF32);
//         - cucc (CPU simulator): CUDA written for the simulator: managed
//           memory instead of copies, packing kernels with streaming stores,
//           and a BLIS-like micro-kernel of 6 x 64 per block of 64 threads;
//   omp:  C++ with OpenMP and AVX-512, BLIS-like (packing, KC/MC/NC cache
//         blocks, 8 x 32 micro-kernel in 16 zmm registers).
// Inputs are small integers, so every product and sum is exact in fp32 (and
// in TF32, whose 10-bit mantissa holds them): every result must equal the
// reference exactly.
//
// Per size: time per kernel (mean of the timed repetitions), total time of
// one multiplication from host data to host result (copies and packing
// included where the contender needs them), and GFLOPS of the kernel.
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

// One block is one row of NR = 64 threads, one column of C per thread, and
// accumulates MR = 6 rows: with 64 lanes a value is 4 zmm registers, so 24 zmm
// of accumulators. Per step of k: 4 vector loads of B and 6 broadcasts of A for
// 24 FMAs (a 12 x 32 tile needs 14 loads for the same 24 FMAs, and memory
// operations cost power, which lowers the clock). The block walks k in passes of
// KC; A and B are packed first so each step reads contiguous memory.
constexpr int SMR = 6, SNR = 64, SKC = 768, SNC = 512;

// One block per (micro-panel of 6 rows, 32 values of k). Lane l of step j writes
// element 32 j + l of the 192 contiguous floats, so the store is one streaming
// (non-temporal) store per step: the packed copy goes to memory without first
// reading the lines it overwrites.
__global__ void __launch_bounds__(32) pack_a(const float* A, float* Ap, int K) {
    const int lane = threadIdx.x, panel = blockIdx.x, k0 = blockIdx.y * 32;
    float* dst = Ap + (long)panel * K * SMR + (long)k0 * SMR;
#pragma unroll
    for (int j = 0; j < SMR; ++j) {
        const int o = 32 * j + lane;
        __stcs(dst + o, A[(long)(panel * SMR + o % SMR) * K + k0 + o / SMR]);
    }
}
// One block per (panel of 64 columns, 256 values of k): each row is a copy of 256 bytes.
__global__ void __launch_bounds__(SNR) pack_b(const float* B, float* Bp, int K, int N) {
    const int lane = threadIdx.x, panel = blockIdx.x, k0 = blockIdx.y * 256;
    const float* src = B + (long)k0 * N + panel * SNR + lane;
    float* dst = Bp + (long)panel * K * SNR + (long)k0 * SNR + lane;
    for (int k = 0; k < 256; ++k) __stcs(dst + k * SNR, src[(long)k * N]);
}
// C[m0 : m0 + mc, n0 : n0 + SNC] = A * B, for mc rows (a multiple of 6) per block.
__global__ void __launch_bounds__(SNR) sgemm(const float* Ap, const float* Bp, float* C, int N, int K, int mc) {
    const int lane = threadIdx.x;
    const int m0 = blockIdx.x * mc, n0 = blockIdx.y * SNC;
    for (int pc = 0; pc < K; pc += SKC)
        for (int jr = 0; jr < SNC; jr += SNR) {
            const float* b = Bp + (long)((n0 + jr) / SNR) * K * SNR + (long)pc * SNR + lane;
            for (int ir = 0; ir < mc; ir += SMR) {
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
// Rows per block: enough blocks to keep the 28 workers busy at small sizes.
static int rows_per_block(int n) { return n <= 1536 ? 24 : 96; }

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

#ifdef __NVCC__
// One multiplication host to host (copies over PCIe included), then `reps`
// kernels timed alone.
static Result run_cuda(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int n, int reps) {
    const size_t bytes = size_t(n) * n * sizeof(float);
    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));
    cublasHandle_t h;
    cublasCreate(&h);
    const float one = 1.0f, zero = 0.0f;
    // Row-major C = A * B is column-major C^T = B^T * A^T.
    auto kernel = [&] {
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, dB, CUDA_R_32F, n, dA, CUDA_R_32F, n, &zero, dC,
                     CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
    };
    // Warm-up (library setup).
    CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
    kernel();
    CHECK(cudaDeviceSynchronize());

    double t0 = now_s();
    CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
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
    cublasDestroy(h);
    cudaFree(dA), cudaFree(dB), cudaFree(dC);
    return {ms / 1e3 / reps, total};
}
#else
// On the simulator the device is the CPU and managed memory is the host's
// memory: an application written for it keeps A, B and C in cudaMallocManaged
// buffers that the host fills and reads directly, with no copies (as the
// OpenMP version works on the host's arrays). Copying the inputs into those
// buffers stands for the application producing them there, so it is not
// timed. Total = packing + multiplication; kernel = the multiplication alone.
static Result run_cuda(const std::vector<float>& hA, const std::vector<float>& hB, std::vector<float>& hC, int n,
                       int reps) {
    const size_t elems = size_t(n) * n, bytes = elems * sizeof(float);
    float *A, *B, *C, *Ap, *Bp;
    CHECK(cudaMallocManaged(&A, bytes));
    CHECK(cudaMallocManaged(&B, bytes));
    CHECK(cudaMallocManaged(&C, bytes));
    CHECK(cudaMalloc(&Ap, bytes));  // packed copies (scratch, like a BLAS workspace)
    CHECK(cudaMalloc(&Bp, bytes));
    std::copy(hA.begin(), hA.end(), A);
    std::copy(hB.begin(), hB.end(), B);
    const int mc = rows_per_block(n);
    auto prepare = [&] {
        pack_a<<<dim3(n / SMR, n / 32), 32>>>(A, Ap, n);
        pack_b<<<dim3(n / SNR, n / 256), SNR>>>(B, Bp, n, n);
    };
    auto kernel = [&] { sgemm<<<dim3(n / mc, n / SNC), SNR>>>(Ap, Bp, C, n, n, mc); };
    // Warm-up (JIT and choice of variant).
    prepare();
    kernel();
    CHECK(cudaDeviceSynchronize());

    double t0 = now_s();
    prepare();
    kernel();
    CHECK(cudaDeviceSynchronize());
    const double total = now_s() - t0;

    t0 = now_s();
    for (int r = 0; r < reps; ++r) kernel();
    CHECK(cudaDeviceSynchronize());
    const double each = (now_s() - t0) / reps;
    CHECK(cudaGetLastError());
    std::copy(C, C + elems, hC.begin());  // for the check only
    cudaFree(A), cudaFree(B), cudaFree(C), cudaFree(Ap), cudaFree(Bp);
    return {each, total};
}
#endif

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
