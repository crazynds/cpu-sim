// 2D heat diffusion (5-point Jacobi stencil) with a shared-memory tile + halo.
//
// Exercises: 2D grids with partial edge blocks, halo loads guarded by
// threadIdx conditions (only edge lanes of the block are active), clamped
// addressing, a boundary condition branch, and many dependent launches.
#include "common.cuh"

constexpr int BX = 16;
constexpr int BY = 16;

__host__ __device__ inline float heat_update(float c, float l, float r, float u, float d,
                                             float alpha) {
    return c + alpha * (l + r + u + d - 4.0f * c);
}

__device__ inline float load_clamped(const float* in, int w, int h, int x, int y) {
    x = min(max(x, 0), w - 1);
    y = min(max(y, 0), h - 1);
    return in[y * w + x];
}

__global__ void heat_step(const float* in, float* out, int w, int h, float alpha) {
    __shared__ float tile[BY + 2][BX + 2];

    int x = blockIdx.x * BX + threadIdx.x;
    int y = blockIdx.y * BY + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    tile[ty][tx] = load_clamped(in, w, h, x, y);
    if (threadIdx.x == 0) tile[ty][0] = load_clamped(in, w, h, x - 1, y);
    if (threadIdx.x == BX - 1) tile[ty][BX + 1] = load_clamped(in, w, h, x + 1, y);
    if (threadIdx.y == 0) tile[0][tx] = load_clamped(in, w, h, x, y - 1);
    if (threadIdx.y == BY - 1) tile[BY + 1][tx] = load_clamped(in, w, h, x, y + 1);
    __syncthreads();

    if (x < w && y < h) {
        float c = tile[ty][tx];
        float v;
        if (x == 0 || y == 0 || x == w - 1 || y == h - 1)
            v = c;  // fixed boundary
        else
            v = heat_update(c, tile[ty][tx - 1], tile[ty][tx + 1], tile[ty - 1][tx],
                            tile[ty + 1][tx], alpha);
        out[y * w + x] = v;
    }
}

int main() {
    const int w = 1000, h = 700, iters = 20;
    const float alpha = 0.2f;

    Lcg rng(5);
    std::vector<float> grid(size_t(w) * h);
    for (auto& v : grid) v = rng.next_float(0.0f, 1.0f);
    for (int y = h / 2 - 50; y < h / 2 + 50; ++y)
        for (int x = w / 2 - 50; x < w / 2 + 50; ++x) grid[size_t(y) * w + x] = 100.0f;

    std::vector<float> ref = grid, next(grid.size());
    for (int it = 0; it < iters; ++it) {
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                size_t i = size_t(y) * w + x;
                if (x == 0 || y == 0 || x == w - 1 || y == h - 1)
                    next[i] = ref[i];
                else
                    next[i] = heat_update(ref[i], ref[i - 1], ref[i + 1], ref[i - w], ref[i + w],
                                          alpha);
            }
        ref.swap(next);
    }

    DeviceBuffer<float> d_a(grid.size()), d_b(grid.size());
    dim3 block(BX, BY);
    dim3 blocks((w + BX - 1) / BX, (h + BY - 1) / BY);

    report_time("heat_step x20", time_ms([&] {
        d_a.upload(grid);
        for (int it = 0; it < iters; ++it) {
            bool even = (it % 2 == 0);
            heat_step<<<blocks, block>>>(even ? d_a.get() : d_b.get(),
                                         even ? d_b.get() : d_a.get(), w, h, alpha);
        }
    }));

    std::vector<float> got;
    (iters % 2 == 0 ? d_a : d_b).download(got);
    bool ok = check_close(ref.data(), got.data(), got.size(), 1e-5, 1e-4, "heat_step");
    return finish("heat2d", ok);
}
