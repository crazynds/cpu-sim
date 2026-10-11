// Bitwise reproducibility: kernels whose CUDA semantics fix every rounding
// (per-thread math, fixed-order shared and warp reductions, stencils, local
// arrays) must give the same bits under every lane width, with and without JIT,
// on both backends and on every run. The program prints one "DIGEST <kernel>
// <hash>" line per kernel; tests/cuda_repro_test.sh compares them across modes.
// It also checks the results against a host model, so a digest that is stable
// but wrong still fails.
#include <cstdint>
#include <vector>

#include "check.cuh"

// FNV-1a over the raw bits; NaNs are canonical so payload bits do not count.
static uint64_t digest(const void* p, size_t bytes, size_t elem = 4) {
    const auto* b = static_cast<const unsigned char*>(p);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; i += elem) {
        unsigned char tmp[8];
        std::memcpy(tmp, b + i, elem);
        if (elem == 4) {
            float f;
            std::memcpy(&f, tmp, 4);
            if (std::isnan(f)) std::memcpy(tmp, "\x00\x00\xc0\x7f", 4);
        } else {
            double d;
            std::memcpy(&d, tmp, 8);
            if (std::isnan(d)) std::memcpy(tmp, "\x00\x00\x00\x00\x00\x00\xf8\x7f", 8);
        }
        for (size_t k = 0; k < elem; ++k) h = (h ^ tmp[k]) * 1099511628211ull;
    }
    return h;
}

static void print_digest(const char* kernel, const void* p, size_t bytes, size_t elem = 4) {
    std::printf("DIGEST %s %016llx\n", kernel, (unsigned long long)digest(p, bytes, elem));
}

// Per-thread math: libmvec functions, IEEE division and sqrt, fma, doubles.
__global__ void math_mix(const float* in, float* out, double* outd, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];
    float a = sinf(x) * cosf(0.5f * x) + expf(-x * x) + logf(1.0f + fabsf(x));
    float b = powf(fabsf(x) + 0.5f, 1.7f) / (1.0f + x * x) + sqrtf(fabsf(x)) - rsqrtf(2.0f + x * x);
    float c = fmaf(a, b, tanhf(x)) + atan2f(x, 1.0f + b) + floorf(x * 3.7f) * 0.25f;
    out[i] = c;
    double xd = x;
    outd[i] = sin(xd) * exp(-0.1 * xd) + sqrt(fabs(xd) + 1.0) / (2.0 + xd * xd) + cbrt(xd);
}

// Tree reduction in shared memory: a fixed summation order per block.
__global__ void block_reduce(const float* in, float* out, int n) {
    extern __shared__ float s[];
    int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
    s[t] = i < n ? in[i] * in[i] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = s[0];
}

// Butterfly reduction across a warp.
__global__ void warp_reduce(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = i < n ? in[i] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    out[i] = v;
}

// 5-point stencil on a 2D grid of 2D blocks, with divergent borders.
__global__ void stencil(const float* in, float* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float c = in[y * w + x];
    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
        out[y * w + x] = c;
        return;
    }
    out[y * w + x] = 0.6f * c + 0.1f * (in[y * w + x - 1] + in[y * w + x + 1] + in[(y - 1) * w + x] + in[(y + 1) * w + x]);
}

// Local array with data-dependent indexing (insertion sort), then a weighted sum.
__global__ void local_sort(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v[8];
    for (int k = 0; k < 8; ++k) v[k] = in[(i * 7 + k * 13) % n];
    for (int k = 1; k < 8; ++k) {
        float key = v[k];
        int j = k - 1;
        while (j >= 0 && v[j] > key) {
            v[j + 1] = v[j];
            --j;
        }
        v[j + 1] = key;
    }
    float acc = 0.0f;
    for (int k = 0; k < 8; ++k) acc = acc * 0.5f + v[k];
    out[i] = acc;
}

int main() {
    Checker check("repro");
    const int n = 50000;  // not a multiple of any block size used
    std::vector<float> in(n);
    uint32_t seed = 12345;
    for (float& v : in) {
        seed = seed * 1664525u + 1013904223u;
        v = (int(seed >> 8) % 20000 - 10000) * 0.0003f;
    }
    float *d_in, *d_out, *d_tmp;
    double* d_outd;
    CHECK_CUDA(cudaMalloc(&d_in, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_tmp, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_outd, n * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_in, in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float> out(n);
    std::vector<double> outd(n);

    // math_mix: block of 96 (3 warps).
    math_mix<<<(n + 95) / 96, 96>>>(d_in, d_out, d_outd, n);
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(outd.data(), d_outd, n * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        float x = in[i];
        float a = std::sin(x) * std::cos(0.5f * x) + std::exp(-x * x) + std::log(1.0f + std::fabs(x));
        float b = std::pow(std::fabs(x) + 0.5f, 1.7f) / (1.0f + x * x) + std::sqrt(std::fabs(x)) - 1.0f / std::sqrt(2.0f + x * x);
        float c = std::fma(a, b, std::tanh(x)) + std::atan2(x, 1.0f + b) + std::floor(x * 3.7f) * 0.25f;
        check.expect(close_rel(out[i], c, 1e-4, 1e-4), "math_mix", i, out[i], c);
        double xd = x;
        double e = std::sin(xd) * std::exp(-0.1 * xd) + std::sqrt(std::fabs(xd) + 1.0) / (2.0 + xd * xd) + std::cbrt(xd);
        check.expect(close_rel(outd[i], e, 1e-12, 1e-12), "math_mix double", i, outd[i], e);
    }
    print_digest("math_mix", out.data(), n * sizeof(float));
    print_digest("math_mix_double", outd.data(), n * sizeof(double), 8);

    // block_reduce: 256 threads, dynamic shared memory.
    const int blocks = (n + 255) / 256;
    block_reduce<<<blocks, 256, 256 * sizeof(float)>>>(d_in, d_tmp, n);
    std::vector<float> partial(blocks);
    CHECK_CUDA(cudaMemcpy(partial.data(), d_tmp, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    for (int b = 0; b < blocks; ++b) {
        float s[256];
        for (int t = 0; t < 256; ++t) s[t] = b * 256 + t < n ? in[b * 256 + t] * in[b * 256 + t] : 0.0f;
        for (int stride = 128; stride > 0; stride >>= 1)
            for (int t = 0; t < stride; ++t) s[t] += s[t + stride];
        check.expect(partial[b] == s[0], "block_reduce (same order: exact)", b, partial[b], s[0]);
    }
    print_digest("block_reduce", partial.data(), blocks * sizeof(float));

    // warp_reduce: the grid covers n rounded up to whole blocks of 128.
    const int wn = (n + 127) / 128 * 128;
    float* d_wout;
    CHECK_CUDA(cudaMalloc(&d_wout, wn * sizeof(float)));
    warp_reduce<<<wn / 128, 128>>>(d_in, d_wout, n);
    std::vector<float> wout(wn);
    CHECK_CUDA(cudaMemcpy(wout.data(), d_wout, wn * sizeof(float), cudaMemcpyDeviceToHost));
    for (int w = 0; w < wn / 32; ++w) {
        float v[32];
        for (int l = 0; l < 32; ++l) v[l] = w * 32 + l < n ? in[w * 32 + l] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            float nv[32];
            for (int l = 0; l < 32; ++l) nv[l] = v[l] + v[l ^ off];
            std::memcpy(v, nv, sizeof v);
        }
        for (int l = 0; l < 32; ++l) check.expect(wout[w * 32 + l] == v[l], "warp_reduce (exact)", w * 32 + l, wout[w * 32 + l], v[l]);
    }
    print_digest("warp_reduce", wout.data(), wn * sizeof(float));

    // stencil: 250 x 200 grid, 16x8 blocks (partial blocks on both edges), 6 steps.
    const int w = 250, h = 200;
    std::vector<float> grid(in.begin(), in.begin() + w * h), next(w * h);
    float *d_a, *d_b;
    CHECK_CUDA(cudaMalloc(&d_a, w * h * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, w * h * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_a, grid.data(), w * h * sizeof(float), cudaMemcpyHostToDevice));
    dim3 sb(16, 8), sg((w + 15) / 16, (h + 7) / 8);
    for (int step = 0; step < 6; ++step) {
        stencil<<<sg, sb>>>(d_a, d_b, w, h);
        std::swap(d_a, d_b);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                float c = grid[y * w + x];
                next[y * w + x] = (x == 0 || y == 0 || x == w - 1 || y == h - 1)
                                      ? c
                                      : 0.6f * c + 0.1f * (grid[y * w + x - 1] + grid[y * w + x + 1] + grid[(y - 1) * w + x] +
                                                           grid[(y + 1) * w + x]);
            }
        std::swap(grid, next);
    }
    std::vector<float> sres(w * h);
    CHECK_CUDA(cudaMemcpy(sres.data(), d_a, w * h * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < w * h; ++i) check.expect(close_rel(sres[i], grid[i], 1e-5, 1e-6), "stencil", i, sres[i], grid[i]);
    print_digest("stencil", sres.data(), w * h * sizeof(float));

    // local_sort: block of 64.
    local_sort<<<(n + 63) / 64, 64>>>(d_in, d_out, n);
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        float v[8];
        for (int k = 0; k < 8; ++k) v[k] = in[(i * 7 + k * 13) % n];
        std::sort(v, v + 8);
        float acc = 0.0f;
        for (int k = 0; k < 8; ++k) acc = acc * 0.5f + v[k];
        check.expect(out[i] == acc, "local_sort (exact)", i, out[i], acc);
    }
    print_digest("local_sort", out.data(), n * sizeof(float));

    cudaFree(d_in), cudaFree(d_out), cudaFree(d_tmp), cudaFree(d_outd), cudaFree(d_wout), cudaFree(d_a), cudaFree(d_b);
    return check.finish();
}
