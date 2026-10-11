// Arithmetic peak of the device, measured through CUDA: a kernel that does
// nothing but independent chains of multiply-adds, on the workers the device
// was configured with. It is the ceiling for a compute-bound kernel at the
// clock the device holds under that load; run it right before a benchmark to
// compare in the same thermal state.
//
//   build/cucc -O3 -std=c++20 bench/peak.cu -o peak
//   ./peak [fp32|bf16|all] [seconds]
//
// fp32: each thread runs 12 chains of fmaf. With 32 lanes per call a float is
// two 512-bit registers, so 24 independent FMA chains per worker, more than
// the FMA latency needs.
// bf16: each warp runs mma_sync on one accumulator fragment kept in registers
// (16 x 16 x 16 per mma; on the CPU, AVX512-BF16 dot products). Needs compute
// capability 8.0; skipped otherwise.
// Also builds with nvcc (-arch=native) for the same measurement on a GPU.
#include <cuda_bf16.h>
#include <mma.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace nvcuda;

#define CHECK(call)                                                                            \
    do {                                                                                       \
        cudaError_t e_ = (call);                                                               \
        if (e_ != cudaSuccess) {                                                               \
            std::printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

constexpr int kChains = 12;

__global__ void __launch_bounds__(32) fp32_chains(float x, float y, long iters, float* out) {
    float acc[kChains];
#pragma unroll
    for (int i = 0; i < kChains; ++i) acc[i] = float(threadIdx.x + i);
    for (long n = 0; n < iters; ++n)
#pragma unroll
        for (int i = 0; i < kChains; ++i) acc[i] = fmaf(x, y, acc[i]);
    float s = 0;
#pragma unroll
    for (int i = 0; i < kChains; ++i) s += acc[i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void __launch_bounds__(32) bf16_mma(const __nv_bfloat16* ab, long iters, float* out) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::load_matrix_sync(a, ab, 16);
    wmma::load_matrix_sync(b, ab + 256, 16);
    wmma::fill_fragment(acc, 0.0f);
    for (long n = 0; n < iters; ++n) wmma::mma_sync(acc, a, b, acc);
    wmma::store_matrix_sync(out + blockIdx.x * 256, acc, 16, wmma::mem_row_major);
}

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Runs `launch(iters)` with the iterations scaled to about `seconds`; returns
// FLOPs per second given the FLOPs of one iteration of the whole grid.
template <typename F>
static double measure(double seconds, double flops_per_iter, F&& launch) {
    long iters = 1000;
    for (;;) {  // calibrate (the first launch also compiles and picks the variant)
        const double t0 = now_s();
        launch(iters);
        CHECK(cudaDeviceSynchronize());
        const double s = now_s() - t0;
        if (s > 0.1) {
            iters = long(iters * seconds / s) + 1;
            break;
        }
        iters *= 4;
    }
    const double t0 = now_s();
    launch(iters);
    CHECK(cudaDeviceSynchronize());
    return flops_per_iter * double(iters) / (now_s() - t0);
}

int main(int argc, char** argv) {
    const std::string which = argc > 1 ? argv[1] : "all";
    const double seconds = argc > 2 ? std::atof(argv[2]) : 2.0;
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    // Enough blocks that every worker (or SM) stays busy to the end.
    const int blocks = p.multiProcessorCount * 64;
    std::printf("%s, compute capability %d.%d, %d multiprocessors, %d blocks of 32 threads\n", p.name, p.major,
                p.minor, p.multiProcessorCount, blocks);
    float* out;
    CHECK(cudaMalloc(&out, size_t(blocks) * 256 * sizeof(float)));

    if (which == "all" || which == "fp32") {
        const double fl = measure(seconds, 2.0 * kChains * 32 * blocks, [&](long iters) {
            fp32_chains<<<blocks, 32>>>(1.0f, 1e-9f, iters, out);
        });
        std::printf("fp32 FMA:        %9.0f GFLOPS\n", fl / 1e9);
    }
    if (which == "all" || which == "bf16") {
        if (p.major * 10 + p.minor < 80) {
            std::printf("bf16 tensor:     skipped (needs compute capability 8.0)\n");
        } else {
            __nv_bfloat16* ab;
            CHECK(cudaMallocManaged(&ab, 512 * sizeof(__nv_bfloat16)));
            for (int i = 0; i < 512; ++i) ab[i] = __float2bfloat16(i < 256 ? 1.0f : 1e-6f);
            const double fl = measure(seconds, 2.0 * 16 * 16 * 16 * blocks, [&](long iters) {
                bf16_mma<<<blocks, 32>>>(ab, iters, out);
            });
            std::printf("bf16 tensor:     %9.0f GFLOPS\n", fl / 1e9);
            CHECK(cudaFree(ab));
        }
    }
    CHECK(cudaFree(out));
}
