// Dense matrix multiplication C = A * B.
//
// Exercises: 2D blocks (a warp spans two rows of a 16x16 block), early return
// (naive), static __shared__ tiles, __syncthreads inside a loop, and bounds
// checks with sizes that are not multiples of the tile.
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

__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int a_col = t + threadIdx.x;
        int b_row = t + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

int main() {
    const int M = 500, N = 470, K = 530;

    Lcg rng(42);
    std::vector<float> A(size_t(M) * K), B(size_t(K) * N);
    for (auto& v : A) v = rng.next_float(-1.0f, 1.0f);
    for (auto& v : B) v = rng.next_float(-1.0f, 1.0f);

    std::vector<float> ref(size_t(M) * N);
    std::vector<double> row_acc(N);
    for (int i = 0; i < M; ++i) {
        std::fill(row_acc.begin(), row_acc.end(), 0.0);
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

    dC.zero();
    report_time("matmul_tiled", time_ms([&] {
        matmul_tiled<<<grid, block>>>(dA.get(), dB.get(), dC.get(), M, N, K);
    }));
    dC.download(C);
    ok &= check_close(ref.data(), C.data(), C.size(), 1e-4, 1e-3, "matmul_tiled");

    return finish("matmul", ok);
}
