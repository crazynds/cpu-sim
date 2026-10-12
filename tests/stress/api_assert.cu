// Device assert that never fires: kernels with assert() on invariants that
// hold must run normally and produce their results.
#include "stress.cuh"

#include <cassert>

__global__ void checked_scale(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    assert(blockDim.x % 32 == 0);
    if (i < n) {
        assert(i >= 0 && i < n);
        float v = in[i];
        assert(v >= -1.0f && v <= 1.0f);
        out[i] = v * 4.0f;
    }
}

__device__ int checked_div(int a, int b) {
    assert(b != 0);
    return a / b;
}

__global__ void divide(const int* a, const int* b, int* q, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) q[i] = checked_div(a[i], b[i]);
}

__global__ void shared_invariant(int* out) {
    __shared__ int s[256];
    s[threadIdx.x] = threadIdx.x;
    __syncthreads();
    int partner = s[(threadIdx.x + 1) % blockDim.x];
    assert(partner == int((threadIdx.x + 1) % blockDim.x));
    out[blockIdx.x * blockDim.x + threadIdx.x] = partner;
}

int main() {
    using namespace stress;

    {
        const int n = 1 << 20;
        auto x = random_floats(n, 1);
        Buffer<float> in(x), out(n);
        double k = time_device([&] { checked_scale<<<(n + 255) / 256, 256>>>(in, out, n); });
        auto got = out.download();
        std::vector<float> want(n);
        for (int i = 0; i < n; ++i) want[i] = x[i] * 4.0f;
        report("assert_holds_scale", equal(got.data(), want.data(), n), k, -1);
    }

    {
        const int n = 1 << 16;
        auto ra = random_u32(n, 2), rb = random_u32(n, 3);
        std::vector<int> a(n), b(n), want(n);
        for (int i = 0; i < n; ++i) {
            a[i] = int(ra[i] % 100000) - 50000;
            b[i] = int(rb[i] % 199) - 99;
            if (b[i] == 0) b[i] = 7;
            want[i] = a[i] / b[i];
        }
        Buffer<int> da(a), db(b), q(n);
        double k = time_device([&] { divide<<<n / 128, 128>>>(da, db, q, n); });
        auto got = q.download();
        report("assert_holds_device_fn", equal(got.data(), want.data(), n), k, -1);
    }

    {
        const int blocks = 64;
        Buffer<int> out(blocks * 256);
        double k = time_device([&] { shared_invariant<<<blocks, 256>>>(out); });
        auto got = out.download();
        bool ok = true;
        for (int i = 0; i < blocks * 256 && ok; ++i) ok = got[i] == (i % 256 + 1) % 256;
        cudaError_t e = cudaGetLastError();
        report("assert_holds_shared", ok && e == cudaSuccess, k, -1);
    }

    return finish();
}
