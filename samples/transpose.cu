// Matrix transpose: naive vs shared-memory tile with padding.
//
// Exercises: the naive kernel reads contiguously but writes with a large
// stride (a scatter for us); the tiled kernel makes both sides contiguous and
// moves the strided access into shared memory ([TT][TT + 1] padding). Each
// thread handles several rows (a uniform inner loop).
#include "common.cuh"

constexpr int TT = 32;
constexpr int BROWS = 8;

__global__ void transpose_naive(const float* in, float* out, int rows, int cols) {
    int x = blockIdx.x * TT + threadIdx.x;
    int y = blockIdx.y * TT + threadIdx.y;
    for (int j = 0; j < TT; j += BROWS)
        if (x < cols && y + j < rows) out[x * rows + (y + j)] = in[(y + j) * cols + x];
}

__global__ void transpose_tiled(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[TT][TT + 1];

    int x = blockIdx.x * TT + threadIdx.x;
    int y = blockIdx.y * TT + threadIdx.y;
    for (int j = 0; j < TT; j += BROWS)
        if (x < cols && y + j < rows) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * cols + x];
    __syncthreads();

    x = blockIdx.y * TT + threadIdx.x;  // now indexes rows of `in`
    y = blockIdx.x * TT + threadIdx.y;  // now indexes cols of `in`
    for (int j = 0; j < TT; j += BROWS)
        if (x < rows && y + j < cols) out[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main() {
    const int rows = SAMPLE_SIZE(24000, 2410), cols = SAMPLE_SIZE(18000, 1830);

    Lcg rng(3);
    std::vector<float> in(size_t(rows) * cols), ref(in.size());
    for (auto& v : in) v = rng.next_float(-100.0f, 100.0f);
#pragma omp parallel for
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) ref[size_t(c) * rows + r] = in[size_t(r) * cols + c];

    DeviceBuffer<float> d_in(in.size()), d_out(in.size());
    d_in.upload(in);
    dim3 block(TT, BROWS);
    dim3 grid((cols + TT - 1) / TT, (rows + TT - 1) / TT);
    std::vector<float> got;
    bool ok = true;

    report_time("transpose_naive", time_ms([&] {
        transpose_naive<<<grid, block>>>(d_in.get(), d_out.get(), rows, cols);
    }));
    d_out.download(got);
    ok &= check_close(ref.data(), got.data(), got.size(), 0.0, 0.0, "transpose_naive");

    d_out.zero();
    report_time("transpose_tiled", time_ms([&] {
        transpose_tiled<<<grid, block>>>(d_in.get(), d_out.get(), rows, cols);
    }));
    d_out.download(got);
    ok &= check_close(ref.data(), got.data(), got.size(), 0.0, 0.0, "transpose_tiled");

    return finish("transpose", ok);
}
