// Repeated launches: the automatic width choice tries different variants over
// the first launches (some blocks of a single launch run with one width, others
// with another) and then settles. The result of every launch must be bitwise
// identical to the first, while the choice changes underneath, for kernels with
// and without barriers, interleaved with each other and with a second block
// shape of the same kernel (a separate choice).
#include <cstdint>
#include <vector>

#include "check.cuh"

__global__ void saxpy_like(const float* x, float* y, int n, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + sinf(x[i]) / (1.0f + y[i] * y[i]);
}

__global__ void tiled_sum(const float* x, float* out, int n) {
    __shared__ float tile[256];
    int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
    tile[t] = i < n ? x[i] : 0.0f;
    __syncthreads();
    float acc = 0.0f;
    for (int k = 0; k < (int)blockDim.x; ++k) acc += tile[(t + k) % blockDim.x] * (k + 1);
    __syncthreads();
    if (i < n) out[i] = acc;
}

int main() {
    Checker check("repeat_launch");
    const int n = 1 << 18, launches = 24;
    std::vector<float> x(n), y0(n);
    for (int i = 0; i < n; ++i) {
        x[i] = (i % 1000) * 0.001f - 0.5f;
        y0[i] = (i % 77) * 0.01f;
    }
    float *d_x, *d_y, *d_t;
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_t, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> first_a, first_b, first_c, got(n);
    for (int l = 0; l < launches; ++l) {
        // saxpy_like from the same initial y each time.
        CHECK_CUDA(cudaMemcpy(d_y, y0.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        saxpy_like<<<n / 256, 256>>>(d_x, d_y, n, 1.5f);
        CHECK_CUDA(cudaMemcpy(got.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
        if (l == 0) first_a = got;
        check.expect(std::memcmp(got.data(), first_a.data(), n * sizeof(float)) == 0, "saxpy_like differs from launch 0", l);

        tiled_sum<<<n / 256, 256>>>(d_x, d_t, n);
        CHECK_CUDA(cudaMemcpy(got.data(), d_t, n * sizeof(float), cudaMemcpyDeviceToHost));
        if (l == 0) first_b = got;
        check.expect(std::memcmp(got.data(), first_b.data(), n * sizeof(float)) == 0, "tiled_sum differs from launch 0", l);

        // Same kernel, other block shape: a separate width choice.
        tiled_sum<<<n / 64, 64>>>(d_x, d_t, n);
        CHECK_CUDA(cudaMemcpy(got.data(), d_t, n * sizeof(float), cudaMemcpyDeviceToHost));
        if (l == 0) first_c = got;
        check.expect(std::memcmp(got.data(), first_c.data(), n * sizeof(float)) == 0, "tiled_sum/64 differs from launch 0", l);
    }

    // And the first launch itself is right.
    for (int i = 0; i < n; ++i) {
        float want = 1.5f * x[i] + std::sin(x[i]) / (1.0f + y0[i] * y0[i]);
        check.expect(close_rel(first_a[i], want, 1e-5, 1e-6), "saxpy_like value", i, first_a[i], want);
    }
    for (int bs : {256, 64}) {
        std::vector<float>& r = bs == 256 ? first_b : first_c;
        for (int i = 0; i < n; i += 97) {
            int b0 = i / bs * bs, t = i - b0;
            float want = 0.0f;
            for (int k = 0; k < bs; ++k) want += x[b0 + (t + k) % bs] * (k + 1);
            check.expect(close_rel(r[i], want, 1e-5, 1e-2), "tiled_sum value (256 terms up to 128: absolute error)", i, r[i], want);
        }
    }
    cudaFree(d_x), cudaFree(d_y), cudaFree(d_t);
    return check.finish();
}
