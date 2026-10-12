// Recoverable errors: each call must return the documented error code, and
// the context must keep working afterwards (none of these is sticky).
#include "stress.cuh"

__global__ void fill(int* p, int n, int v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

__global__ void uses_dyn_smem(int* p) {
    extern __shared__ int s[];
    s[threadIdx.x] = threadIdx.x;
    __syncthreads();
    p[threadIdx.x] = s[threadIdx.x];
}

// Launches fill and checks the result: the context still works.
static bool context_works(int v) {
    const int n = 4096;
    int* d = nullptr;
    if (cudaMalloc(&d, n * sizeof(int)) != cudaSuccess) return false;
    fill<<<n / 256, 256>>>(d, n, v);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return false;
    std::vector<int> h(n);
    if (cudaMemcpy(h.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    cudaFree(d);
    for (int x : h)
        if (x != v) return false;
    return true;
}

// A bad launch configuration: CUDA up to 12 reports cudaErrorInvalidConfiguration,
// CUDA 13 on the RTX 5060 reports cudaErrorInvalidValue. Either is accepted.
static bool bad_config(cudaError_t e, const char* what) {
    bool ok = e == cudaErrorInvalidConfiguration || e == cudaErrorInvalidValue;
    if (!ok) std::printf("  %s: got %d (%s), want a configuration error\n", what, int(e), cudaGetErrorName(e));
    return ok;
}

static bool expect(cudaError_t got, cudaError_t want, const char* what) {
    if (got != want) std::printf("  %s: got %d (%s), want %d (%s)\n", what, int(got), cudaGetErrorName(got), int(want),
                                 cudaGetErrorName(want));
    return got == want;
}

int main() {
    using namespace stress;

    {  // 1025 threads per block
        int* d;
        STRESS_CHECK(cudaMalloc(&d, 2048 * sizeof(int)));
        fill<<<1, 1025>>>(d, 2048, 1);
        cudaError_t e = cudaGetLastError();
        cudaError_t e2 = cudaGetLastError();  // cleared by the first read
        bool ok = bad_config(e, "1025 threads") && e2 == cudaSuccess;
        // Also an illegal grid/block shape: blockDim.z > 64.
        fill<<<1, dim3(1, 1, 65)>>>(d, 2048, 1);
        ok = bad_config(cudaGetLastError(), "block z 65") && ok;
        // And a zero-sized grid.
        fill<<<0, 32>>>(d, 2048, 1);
        ok = bad_config(cudaGetLastError(), "grid 0") && ok;
        fill<<<2, 1024>>>(d, 2048, 5);
        ok = ok && cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
        std::vector<int> h(2048);
        STRESS_CHECK(cudaMemcpy(h.data(), d, 2048 * sizeof(int), cudaMemcpyDeviceToHost));
        for (int x : h) ok = ok && x == 5;
        STRESS_CHECK(cudaFree(d));
        report("launch_too_many_threads", ok && context_works(11), 0, -1);
    }

    {  // dynamic shared memory over the default limit without the opt-in attribute
        int* d;
        STRESS_CHECK(cudaMalloc(&d, 64 * sizeof(int)));
        uses_dyn_smem<<<1, 64, 200 * 1024>>>(d);  // more than any opt-in limit
        cudaError_t e = cudaGetLastError();
        bool ok = e != cudaSuccess;
        if (!ok) std::printf("  200 KiB dynamic smem launch succeeded\n");
        uses_dyn_smem<<<1, 64, 64 * sizeof(int)>>>(d);
        ok = ok && cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
        STRESS_CHECK(cudaFree(d));
        report("launch_too_much_smem", ok && context_works(12), 0, -1);
    }

    {  // absurd allocation
        void* p = reinterpret_cast<void*>(0x1);
        cudaError_t e = cudaMalloc(&p, size_t(1) << 50);  // 1 PiB
        bool ok = expect(e, cudaErrorMemoryAllocation, "cudaMalloc 1 PiB");
        ok = expect(cudaGetLastError(), cudaErrorMemoryAllocation, "last error") && ok;
        ok = expect(cudaGetLastError(), cudaSuccess, "cleared") && ok;
        void* m = nullptr;
        ok = expect(cudaMallocManaged(&m, size_t(1) << 50), cudaErrorMemoryAllocation, "cudaMallocManaged 1 PiB") && ok;
        cudaGetLastError();
        void* q = nullptr;
        ok = ok && cudaMalloc(&q, 1 << 20) == cudaSuccess && q != nullptr;
        cudaFree(q);
        report("malloc_absurd_size", ok && context_works(13), 0, -1);
    }

    {  // cudaFree of a pointer that cudaMalloc never returned
        int* d;
        STRESS_CHECK(cudaMalloc(&d, 4096));
        int on_stack = 0;
        cudaError_t e1 = cudaFree(d + 1);           // inside an allocation, not its start
        cudaGetLastError();
        cudaError_t e2 = cudaFree(&on_stack);       // a host pointer
        cudaGetLastError();
        cudaError_t e3 = cudaFree(nullptr);         // explicitly allowed
        bool ok = expect(e1, cudaErrorInvalidValue, "free interior") && expect(e2, cudaErrorInvalidValue, "free host") &&
                  expect(e3, cudaSuccess, "free null");
        ok = expect(cudaFree(d), cudaSuccess, "free real") && ok;
        cudaError_t e4 = cudaFreeHost(&on_stack);   // not page-locked
        ok = (e4 != cudaSuccess) && ok;
        cudaGetLastError();
        report("free_invalid_pointer", ok && context_works(14), 0, -1);
    }

    {  // cudaMemcpy / cudaMemset with a null pointer
        int* d;
        STRESS_CHECK(cudaMalloc(&d, 4096));
        std::vector<int> h(1024, 3);
        cudaError_t e1 = cudaMemcpy(nullptr, h.data(), 4096, cudaMemcpyHostToDevice);
        cudaGetLastError();
        cudaError_t e2 = cudaMemcpy(h.data(), nullptr, 4096, cudaMemcpyDeviceToHost);
        cudaGetLastError();
        cudaError_t e3 = cudaMemset(nullptr, 0, 4096);
        cudaGetLastError();
        cudaError_t e4 = cudaMemcpy(d, h.data(), 0, cudaMemcpyHostToDevice);  // zero bytes is fine
        bool ok = expect(e1, cudaErrorInvalidValue, "memcpy dst null") && expect(e2, cudaErrorInvalidValue, "memcpy src null") &&
                  expect(e3, cudaErrorInvalidValue, "memset null") && expect(e4, cudaSuccess, "memcpy 0 bytes");
        ok = ok && cudaMemcpy(d, h.data(), 4096, cudaMemcpyHostToDevice) == cudaSuccess;
        STRESS_CHECK(cudaFree(d));
        report("memcpy_null_pointer", ok && context_works(15), 0, -1);
    }

    {  // invalid handles and arguments in other calls
        int dev_count = 0;
        STRESS_CHECK(cudaGetDeviceCount(&dev_count));
        bool ok = expect(cudaSetDevice(dev_count + 5), cudaErrorInvalidDevice, "set bad device");
        cudaGetLastError();
        cudaDeviceProp p;
        ok = expect(cudaGetDeviceProperties(&p, -1), cudaErrorInvalidDevice, "props bad device") && ok;
        cudaGetLastError();
        int v = 0;
        ok = expect(cudaDeviceGetAttribute(&v, cudaDevAttrWarpSize, 99), cudaErrorInvalidDevice, "attr bad device") && ok;
        cudaGetLastError();
        int* d;
        STRESS_CHECK(cudaMalloc(&d, 1024));
        ok = expect(cudaMemcpy(d, d, 1024, cudaMemcpyKind(42)), cudaErrorInvalidMemcpyDirection, "bad kind") && ok;
        cudaGetLastError();
        STRESS_CHECK(cudaFree(d));
        std::printf("  error string: %s / %s\n", cudaGetErrorName(cudaErrorInvalidValue), cudaGetErrorString(cudaErrorInvalidValue));
        ok = ok && std::strcmp(cudaGetErrorName(cudaErrorInvalidConfiguration), "cudaErrorInvalidConfiguration") == 0 &&
             std::strcmp(cudaGetErrorName(cudaSuccess), "cudaSuccess") == 0;
        ok = ok && cudaPeekAtLastError() == cudaSuccess;
        report("invalid_arguments", ok && context_works(16), 0, -1);
    }

    {  // cudaPeekAtLastError keeps the error, cudaGetLastError clears it
        int* d;
        STRESS_CHECK(cudaMalloc(&d, 4096));
        fill<<<1, 2048>>>(d, 1024, 1);
        cudaError_t p1 = cudaPeekAtLastError();
        cudaError_t p2 = cudaPeekAtLastError();
        cudaError_t g1 = cudaGetLastError();
        cudaError_t g2 = cudaGetLastError();
        bool ok = bad_config(p1, "2048 threads") && p2 == p1 && g1 == p1 && g2 == cudaSuccess;
        // A configuration error does not poison synchronization.
        ok = ok && cudaDeviceSynchronize() == cudaSuccess;
        STRESS_CHECK(cudaFree(d));
        report("peek_vs_get_last_error", ok && context_works(17), 0, -1);
    }

    return finish();
}
