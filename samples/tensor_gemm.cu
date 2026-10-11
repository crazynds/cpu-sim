// Tensor cores through nvcuda::wmma: C = A * B for bf16 and fp16 (fp32
// accumulators) and int8 (int32 accumulators), reporting TFLOPS (TOPS for
// int8).
//
// Like a CUDA program, it first checks the device's compute capability: fp16
// tensor cores need 7.0, int8 7.2, bf16 8.0. On the CPU simulator the compute
// capability comes from the dot product instructions the CPU has
// (AVX512-BF16 and AVX512-VNNI; `cpu-smi -q` shows them), and a type it does
// not support is skipped here, as the launch would fail.
//
// Each block of 4 warps computes a 64 x 64 tile of C; each warp a 32 x 32
// part, as 2 x 2 accumulator fragments. A and B go through shared memory as
// 16 x 16 tiles with rows padded to 32 values, so every fragment pointer is
// 256-bit aligned (what the GPU requires, int8 included) and each thread
// copies runs of 16 consecutive values.
//
//   tensor_gemm [bf16|fp16|int8|all] [size] [iterations]
#include <mma.h>

#include <cstring>
#include <string>

#include "common.cuh"

using namespace nvcuda;

constexpr int BM = 64, BN = 64, BK = 32;  // block tile; k in two 16-wide slices
constexpr int LDS = 32;                   // shared row: 16 values of k + 16 of padding

template <typename T>
struct Traits;
template <>
struct Traits<__nv_bfloat16> {
    using Acc = float;
    static constexpr const char* name = "bf16 -> fp32";
    static constexpr int min_cc = 80;
    static __nv_bfloat16 from(int v) { return __float2bfloat16(float(v)); }
};
template <>
struct Traits<__half> {
    using Acc = float;
    static constexpr const char* name = "fp16 -> fp32";
    static constexpr int min_cc = 70;
    static __half from(int v) { return __float2half(float(v)); }
};
template <>
struct Traits<signed char> {
    using Acc = int;
    static constexpr const char* name = "int8 -> int32";
    static constexpr int min_cc = 72;
    static signed char from(int v) { return (signed char)v; }
};

template <typename T>
__global__ void __launch_bounds__(128) wmma_gemm(const T* A, const T* B, typename Traits<T>::Acc* C, int M, int N, int K) {
    using Acc = typename Traits<T>::Acc;
    __shared__ __align__(32) T As[2][BM][LDS];           // [k slice][m][k]
    __shared__ __align__(32) T Bs[2][BN / 16][16][LDS];  // [k slice][16-column group][k][n]
    const int warp = threadIdx.x / 32, wm = warp / 2, wn = warp % 2;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, Acc> acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0);

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int s = 0; s < 2; ++s)
            for (int i = 0; i < BM * 16 / 128; ++i) {
                const int e = i * 128 + threadIdx.x, m = e / 16, k = e % 16;
                As[s][m][k] = A[size_t(row0 + m) * K + k0 + s * 16 + k];
            }
        for (int s = 0; s < 2; ++s)
            for (int i = 0; i < 16 * BN / 128; ++i) {
                const int e = i * 128 + threadIdx.x, k = e / BN, n = e % BN;
                Bs[s][n / 16][k][n % 16] = B[size_t(k0 + s * 16 + k) * N + col0 + n];
            }
        __syncthreads();
        for (int s = 0; s < 2; ++s) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b[2];
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &As[s][wm * 32 + i * 16][0], LDS);
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &Bs[s][wn * 2 + j][0][0], LDS);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(C + size_t(row0 + wm * 32 + i * 16) * N + col0 + wn * 32 + j * 16, acc[i][j], N,
                                    wmma::mem_row_major);
}

// Runs one type; false if a checked value is wrong.
template <typename T>
bool run(int cc, int size, int iterations) {
    using Acc = typename Traits<T>::Acc;
    const char* name = Traits<T>::name;
    if (cc < Traits<T>::min_cc) {
        std::printf("  %-14s skipped: needs compute capability %d.%d\n", name, Traits<T>::min_cc / 10,
                    Traits<T>::min_cc % 10);
        return true;
    }
    const int M = size, N = size, K = size;
    // Small integers: every product and sum is exact in the accumulator, so
    // the result must equal the reference.
    Lcg rng(7);
    std::vector<int> a(size_t(M) * K), b(size_t(K) * N);
    for (auto& v : a) v = int(rng.next_u32() % 5) - 2;
    for (auto& v : b) v = int(rng.next_u32() % 5) - 2;
    std::vector<T> ha(a.size()), hb(b.size());
    for (size_t i = 0; i < a.size(); ++i) ha[i] = Traits<T>::from(a[i]);
    for (size_t i = 0; i < b.size(); ++i) hb[i] = Traits<T>::from(b[i]);

    DeviceBuffer<T> dA(ha.size()), dB(hb.size());
    DeviceBuffer<Acc> dC(size_t(M) * N);
    dA.upload(ha);
    dB.upload(hb);
    const dim3 grid(N / BN, M / BM), block(128);
    wmma_gemm<T><<<grid, block>>>(dA.get(), dB.get(), dC.get(), M, N, K);  // warm-up (and JIT on the CPU)
    CUDA_CHECK_KERNEL();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iterations; ++it) wmma_gemm<T><<<grid, block>>>(dA.get(), dB.get(), dC.get(), M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK_KERNEL();
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= float(iterations);
    const double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;

    // Check 64 rows spread over C (the full reference would take minutes on one core).
    std::vector<Acc> c;
    dC.download(c);
    size_t bad = 0;
    for (int r = 0; r < 64; ++r) {
        const int i = int((size_t(r) * 2654435761u) % unsigned(M));
        std::vector<long long> row(N, 0);
        for (int k = 0; k < K; ++k) {
            const long long av = a[size_t(i) * K + k];
            for (int j = 0; j < N; ++j) row[j] += av * b[size_t(k) * N + j];
        }
        for (int j = 0; j < N; ++j) bad += (long long)c[size_t(i) * N + j] != row[j];
    }
    std::printf("  %-14s %5d x %5d x %5d  %9.3f ms  %8.2f %s  %s\n", name, M, N, K, ms, tflops,
                std::is_same_v<T, signed char> ? "TOPS  " : "TFLOPS", bad ? "FAIL" : "ok");
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return bad == 0;
}

int main(int argc, char** argv) {
    const std::string which = argc > 1 ? argv[1] : "all";
    const int size = argc > 2 ? std::atoi(argv[2]) : SAMPLE_SIZE(4096, 256);
    const int iterations = argc > 3 ? std::atoi(argv[3]) : SAMPLE_SIZE(10, 2);
    if (size <= 0 || size % 64 != 0) {
        std::fprintf(stderr, "size must be a positive multiple of 64\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int cc = prop.major * 10 + prop.minor;
    std::printf("Device: %s, compute capability %d.%d\n", prop.name, prop.major, prop.minor);
    std::printf("Tensor cores: fp16 %s, int8 %s, bf16 %s\n", cc >= 70 ? "yes" : "no", cc >= 72 ? "yes" : "no",
                cc >= 80 ? "yes" : "no");
    if (cc < 70) {
        std::printf("No tensor cores on this device.\n");
        return finish("tensor_gemm", true);
    }

    bool ok = true;
    if (which == "all" || which == "bf16") ok &= run<__nv_bfloat16>(cc, size, iterations);
    if (which == "all" || which == "fp16") ok &= run<__half>(cc, size, iterations);
    if (which == "all" || which == "int8") ok &= run<signed char>(cc, size, iterations);
    return finish("tensor_gemm", ok);
}
