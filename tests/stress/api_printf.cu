// Device printf: a few lines with assorted formats. The check is only that
// the kernels complete and report success (the output order between threads
// is unspecified); the buffer is flushed by the synchronize.
#include "stress.cuh"

__global__ void hello(int tag) {
    if (threadIdx.x == 0) printf("hello from block %d of %d, tag %d\n", blockIdx.x, gridDim.x, tag);
}

__global__ void formats(const float* v, const char* name) {
    int t = threadIdx.x;
    if (t < 3)
        printf("[%s] t=%d f=%.3f e=%e d=%lf u=%u x=%#x ll=%lld c=%c s=%s %%\n", name, t, v[t], v[t] * 1e6f,
               double(v[t]) / 3.0, 4000000000u, 0xbeefu + t, -1234567890123ll, 'A' + t, t == 1 ? "one" : "other");
}

__global__ void conditional_print(const int* data, int n, int* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && data[i] < 0) {
        atomicAdd(count, 1);
        printf("negative at %d: %d (warp lane %d)\n", i, data[i], threadIdx.x % 32);
    }
}

int main() {
    using namespace stress;

    {
        double k = time_device([&] { hello<<<2, 64>>>(1); }, 1);
        report("printf_hello", true, k, -1);
    }

    {
        std::vector<float> v = {1.5f, -2.25f, 3.0e-3f};
        Buffer<float> dv(v);
        char* dn;
        const char name[] = "fmt";
        STRESS_CHECK(cudaMalloc(&dn, sizeof(name)));
        STRESS_CHECK(cudaMemcpy(dn, name, sizeof(name), cudaMemcpyHostToDevice));
        double t0 = now_ms();
        formats<<<1, 32>>>(dv, dn);
        cudaError_t e1 = cudaGetLastError();
        cudaError_t e2 = cudaDeviceSynchronize();
        double k = now_ms() - t0;
        STRESS_CHECK(cudaFree(dn));
        report("printf_formats", e1 == cudaSuccess && e2 == cudaSuccess, k, -1);
    }

    {  // a few threads out of many print; the kernel's other work still checked
        const int n = 1 << 16;
        std::vector<int> d(n);
        for (int i = 0; i < n; ++i) d[i] = (i == 5 || i == 40000 || i == 65535) ? -i : i;
        Buffer<int> dd(d), count(1);
        count.zero();
        double t0 = now_ms();
        conditional_print<<<n / 256, 256>>>(dd, n, count);
        cudaError_t e1 = cudaGetLastError();
        cudaError_t e2 = cudaDeviceSynchronize();
        double k = now_ms() - t0;
        int c = count.download()[0];
        report("printf_conditional", e1 == cudaSuccess && e2 == cudaSuccess && c == 3, k, -1);
    }

    return finish();
}
