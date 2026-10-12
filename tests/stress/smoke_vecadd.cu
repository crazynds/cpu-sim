// Smoke test of the harness: vector add and saxpy.
#include "stress.cuh"

__global__ void vecadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    using namespace stress;
    const int n = 1 << 24;
    auto a = random_floats(n, 1), b = random_floats(n, 2);
    std::vector<float> want(n);
    double ref = best_ms([&] {
#pragma omp parallel for
        for (int i = 0; i < n; ++i) want[i] = a[i] + b[i];
    });
    Buffer<float> da(a), db(b), dc(n);
    double k = time_device([&] { vecadd<<<(n + 255) / 256, 256>>>(da, db, dc, n); });
    auto got = dc.download();
    report("vecadd", equal(got.data(), want.data(), n), k, ref);
    return finish();
}
