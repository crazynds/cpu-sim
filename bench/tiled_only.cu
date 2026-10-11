#define main sample_main
#include "../samples/matmul.cu"
#undef main
#include "bench.cuh"
#include <cstring>

// One kernel only, for perf stat: bench/tiled_only <naive|tiled>
int main(int argc, char** argv) {
    const int M = 4100, N = 4070, K = 4130;
    auto A = random_floats<float>(size_t(M) * K, 1, -1, 1), B = random_floats<float>(size_t(K) * N, 2, -1, 1);
    DeviceBuffer<float> dA(A.size()), dB(B.size()), dC(size_t(M) * N);
    dA.upload(A);
    dB.upload(B);
    dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    bool tiled = argc > 1 && std::strcmp(argv[1], "tiled") == 0;
    if (tiled) bench_report("matmul_tiled", bench_ms([&] { matmul_tiled<<<grid, block>>>(dA.get(), dB.get(), dC.get(), M, N, K); }, 3));
    else bench_report("matmul_naive", bench_ms([&] { matmul_naive<<<grid, block>>>(dA.get(), dB.get(), dC.get(), M, N, K); }, 3));
}
