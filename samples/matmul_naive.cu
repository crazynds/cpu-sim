// Dense matrix multiplication C = A * B, naive kernel only.
//
// The first target of the CPU SIMT compiler (proof of concept): 2D blocks (a
// warp spans two rows of a 16x16 block), early return, a uniform loop inside a
// divergent region, and sizes that are not multiples of the block.
#include "common.cuh"

constexpr int TILE = 16;

__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

int main() {
    const int M = SAMPLE_SIZE(4100, 1030), N = SAMPLE_SIZE(4070, 1010), K = SAMPLE_SIZE(4130, 1050);

    Lcg rng(42);
    std::vector<float> A(size_t(M) * K), B(size_t(K) * N);
    for (auto& v : A) v = rng.next_float(-1.0f, 1.0f);
    for (auto& v : B) v = rng.next_float(-1.0f, 1.0f);

    std::vector<float> ref(size_t(M) * N);
#pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < M; ++i) {
        std::vector<double> row_acc(N, 0.0);
        for (int k = 0; k < K; ++k) {
            double a = A[size_t(i) * K + k];
            for (int j = 0; j < N; ++j) row_acc[j] += a * B[size_t(k) * N + j];
        }
        for (int j = 0; j < N; ++j) ref[size_t(i) * N + j] = static_cast<float>(row_acc[j]);
    }

    DeviceBuffer<float> dA(A.size()), dB(B.size()), dC(ref.size());
    dA.upload(A);
    dB.upload(B);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    std::vector<float> C;
    bool ok = true;

    report_time("matmul_naive", time_ms([&] {
        matmul_naive<<<grid, block>>>(dA.get(), dB.get(), dC.get(), M, N, K);
    }));
    dC.download(C);
    ok &= check_close(ref.data(), C.data(), C.size(), 1e-4, 1e-3, "matmul_naive");

    return finish("matmul_naive", ok);
}
