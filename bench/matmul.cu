#define main sample_main
#include "../samples/matmul.cu"
#undef main
#include "bench.cuh"

int main() {
    const int M = 4100, N = 4070, K = 4130;  // samples/matmul.cu
    auto A = random_floats<float>(size_t(M) * K, 1, -1, 1), B = random_floats<float>(size_t(K) * N, 2, -1, 1);
    DeviceBuffer<float> dA(A.size()), dB(B.size()), dC(size_t(M) * N);
    dA.upload(A);
    dB.upload(B);
    dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    bench_report("matmul_naive", bench_ms([&] { matmul_naive<<<grid, block>>>(dA.get(), dB.get(), dC.get(), M, N, K); }));
    bench_checksum("matmul_naive", dC);
    bench_report("matmul_tiled", bench_ms([&] { matmul_tiled<<<grid, block>>>(dA.get(), dB.get(), dC.get(), M, N, K); }));
    bench_checksum("matmul_tiled", dC);
}
