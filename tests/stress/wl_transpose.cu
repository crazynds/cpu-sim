// Matrix transpose in the style of the NVIDIA transpose sample: 32x32 tiles
// processed by 32x8 thread blocks (TILE_DIM 32, BLOCK_ROWS 8). Cases: a copy
// kernel baseline with the same access pattern, the naive transpose, the
// shared-memory tiled transpose without padding (bank conflicts), the tiled
// transpose with a [TILE_DIM][TILE_DIM+1] tile, a 32x32-thread variant (one
// element per thread), a double-precision variant, and a non-square size that
// is not a multiple of 32 (boundary checks). All exact.
#include "stress.cuh"

#define TILE_DIM 32
#define BLOCK_ROWS 8

// Baseline: coalesced copy with the transpose's indexing pattern.
__global__ void copy_kernel(float* __restrict__ odata, const float* __restrict__ idata, int width, int height) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) odata[(y + j) * width + x] = idata[(y + j) * width + x];
}

// Naive: coalesced reads, strided writes.
__global__ void transpose_naive(float* __restrict__ odata, const float* __restrict__ idata, int width, int height) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) odata[x * height + (y + j)] = idata[(y + j) * width + x];
}

// Tiled, no padding: coalesced on both sides, 32-way bank conflicts on the
// column reads of the tile.
__global__ void transpose_coalesced(float* __restrict__ odata, const float* __restrict__ idata, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
}

// Tiled with padding against bank conflicts.
template <typename T>
__global__ void transpose_no_bank_conflicts(T* __restrict__ odata, const T* __restrict__ idata, int width, int height) {
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
}

// 32x32 threads, one element per thread, padded tile.
__global__ void transpose_tile_32x32(float* __restrict__ odata, const float* __restrict__ idata, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = idata[y * width + x];
    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    odata[y * height + x] = tile[threadIdx.x][threadIdx.y];
}

// Padded tile with bounds checks, for any width and height.
__global__ void transpose_bounded(float* __restrict__ odata, const float* __restrict__ idata, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        if (x < width && y + j < height) tile[threadIdx.y + j][threadIdx.x] = idata[(size_t)(y + j) * width + x];
    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // column of the output = row of the input
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        if (x < height && y + j < width) odata[(size_t)(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
}

// Host references (blocked, all threads).
template <typename T>
static void ref_transpose(const T* in, T* out, int width, int height) {
    const int B = 32;
#pragma omp parallel for collapse(2) schedule(static)
    for (int by = 0; by < height; by += B)
        for (int bx = 0; bx < width; bx += B) {
            int ye = std::min(by + B, height), xe = std::min(bx + B, width);
            for (int x = bx; x < xe; ++x)
#pragma omp simd
                for (int y = by; y < ye; ++y) out[(size_t)x * height + y] = in[(size_t)y * width + x];
        }
}

static void ref_copy(const float* in, float* out, size_t n) {
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; ++i) out[i] = in[i];
}

int main() {
    using namespace stress;
    const int W = 8192, H = 4096;  // width, height (multiples of TILE_DIM)
    const size_t N = (size_t)W * H;
    auto in = random_floats(N, 21);
    std::vector<float> want(N), want_copy(N);
    double ref_t = best_ms([&] { ref_transpose(in.data(), want.data(), W, H); });
    double ref_c = best_ms([&] { ref_copy(in.data(), want_copy.data(), N); });
    Buffer<float> din(in), dout(N);
    dim3 grid(W / TILE_DIM, H / TILE_DIM), threads(TILE_DIM, BLOCK_ROWS);

    auto check = [&](const char* name, const std::vector<float>& ref, double ref_ms, const std::function<void()>& launch) {
        dout.zero();
        double k = time_device(launch);
        auto got = dout.download();
        report(name, equal(got.data(), ref.data(), N, name), k, ref_ms);
    };

    check("copy_baseline", want_copy, ref_c, [&] { copy_kernel<<<grid, threads>>>(dout, din, W, H); });
    check("transpose_naive", want, ref_t, [&] { transpose_naive<<<grid, threads>>>(dout, din, W, H); });
    check("transpose_tiled_nopad", want, ref_t, [&] { transpose_coalesced<<<grid, threads>>>(dout, din, W, H); });
    check("transpose_tiled_padded", want, ref_t,
          [&] { transpose_no_bank_conflicts<float><<<grid, threads>>>(dout, din, W, H); });
    check("transpose_tile_32x32_threads", want, ref_t,
          [&] { transpose_tile_32x32<<<grid, dim3(TILE_DIM, TILE_DIM)>>>(dout, din, W, H); });

    // Double precision, padded tile.
    {
        const int w = 4096, h = 4096;
        const size_t n = (size_t)w * h;
        std::vector<double> a(n);
        for (size_t i = 0; i < n; ++i) a[i] = double(in[i]) * 3.0 + 1e-9 * double(i);
        std::vector<double> wd(n);
        double ref = best_ms([&] { ref_transpose(a.data(), wd.data(), w, h); });
        Buffer<double> da(a), db(n);
        double k = time_device(
            [&] { transpose_no_bank_conflicts<double><<<dim3(w / TILE_DIM, h / TILE_DIM), threads>>>(db, da, w, h); });
        auto got = db.download();
        report("transpose_padded_f64", equal(got.data(), wd.data(), n, "f64"), k, ref);
    }

    // Non-square, not a multiple of 32.
    {
        const int w = 7001, h = 4003;
        const size_t n = (size_t)w * h;
        auto a = random_floats(n, 22);
        std::vector<float> wr(n);
        double ref = best_ms([&] { ref_transpose(a.data(), wr.data(), w, h); });
        Buffer<float> da(a), db(n);
        dim3 g((w + TILE_DIM - 1) / TILE_DIM, (h + TILE_DIM - 1) / TILE_DIM);
        double k = time_device([&] { transpose_bounded<<<g, threads>>>(db, da, w, h); });
        auto got = db.download();
        report("transpose_rect_7001x4003", equal(got.data(), wr.data(), n, "rect"), k, ref);
    }

    return finish();
}
