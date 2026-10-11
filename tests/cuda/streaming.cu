// Streaming stores (__stcs): non-temporal where a whole chunk of 16 lanes
// writes 64 aligned bytes, ordinary stores elsewhere. Checks every case, and
// that the next kernel (on other workers) sees the data: non-temporal stores
// are weakly ordered and need the sfence the compiler adds at the block's end.
#include <vector>

#include "check.cuh"

// Contiguous, aligned, full warps: the non-temporal path.
__global__ void copy(const float* in, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) __stcs(out + i, in[i]);
}
// The address is computed from an index the compiler cannot split into a base
// pointer and offsets (regression: such a store was dropped by LLVM).
__global__ void interleave(const float* in, float* out, int rows, int cols) {
    const int lane = threadIdx.x, panel = blockIdx.x, k0 = blockIdx.y * 32;
    float* dst = out + (long)panel * cols * 6 + (long)k0 * 6;
#pragma unroll
    for (int j = 0; j < 6; ++j) {
        const int o = 32 * j + lane, m = panel * 6 + o % 6;
        __stcs(dst + o, m < rows ? in[(long)m * cols + k0 + o / 6] : 0.0f);
    }
}
// Strided and partial: the masked or scattered path.
__global__ void strided(const int* in, int* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) __stcs(out + 3 * i, in[i] + 1);
}
__global__ void sum_next(const float* in, double* out, int n) {
    double s = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += in[i];
    out[threadIdx.x] = s;
}

int main() {
    Checker c("streaming");
    const int n = 1 << 20;
    float *a, *b;
    double* sums;
    CHECK_CUDA(cudaMallocManaged(&a, n * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&b, n * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&sums, 64 * sizeof(double)));
    for (int i = 0; i < n; ++i) a[i] = float(i % 1000);

    for (int len : {n, n - 7}) {  // the second leaves a partial last warp
        std::fill(b, b + n, -1.0f);
        copy<<<(len + 255) / 256, 256>>>(a, b, len);
        sum_next<<<1, 64>>>(b, sums, len);
        CHECK_CUDA(cudaDeviceSynchronize());
        double got = 0, want = 0;
        for (int t = 0; t < 64; ++t) got += sums[t];
        for (int i = 0; i < len; ++i) want += a[i];
        c.expect(got == want, "copy then read by the next kernel", len, got, want);
        for (int i = 0; i < n; ++i) c.expect(b[i] == (i < len ? a[i] : -1.0f), "copy", i, b[i], i < len ? a[i] : -1);
    }

    const int rows = 100, cols = 64, panels = (rows + 5) / 6;
    std::vector<float> want(size_t(panels) * 6 * cols);
    float* p;
    CHECK_CUDA(cudaMallocManaged(&p, want.size() * sizeof(float)));
    interleave<<<dim3(panels, cols / 32), 32>>>(a, p, rows, cols);
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int m = 0; m < panels * 6; ++m)
        for (int k = 0; k < cols; ++k) {
            const size_t at = size_t(m / 6) * cols * 6 + size_t(k) * 6 + m % 6;
            const float w = m < rows ? a[size_t(m) * cols + k] : 0.0f;
            c.expect(p[at] == w, "interleave", (long long)at, p[at], w);
        }

    int *x, *y;
    const int m = 1000;
    CHECK_CUDA(cudaMallocManaged(&x, m * sizeof(int)));
    CHECK_CUDA(cudaMallocManaged(&y, 3 * m * sizeof(int)));
    for (int i = 0; i < m; ++i) x[i] = i;
    std::fill(y, y + 3 * m, 0);
    strided<<<(m + 127) / 128, 128>>>(x, y, m);
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int i = 0; i < 3 * m; ++i) c.expect(y[i] == (i % 3 ? 0 : i / 3 + 1), "strided", i, y[i], i % 3 ? 0 : i / 3 + 1);
    return c.finish();
}
