// Exercises the runtime API; runs once per backend (see CMakeLists.txt).
#include <cstdint>
#include <cstring>
#include <vector>

#include "check.h"
#include "cuda_runtime.h"

int main() {
    int count = 0;
    EXPECT(cudaGetDeviceCount(&count) == cudaSuccess && count == 1);
    cudaDeviceProp p;
    EXPECT(cudaGetDeviceProperties(&p, 0) == cudaSuccess);
    EXPECT(p.warpSize == 32 && p.major == 12 && p.multiProcessorCount > 0);
    EXPECT(cudaGetDeviceProperties(&p, 1) == cudaErrorInvalidDevice);
    EXPECT(cudaGetLastError() == cudaErrorInvalidDevice);
    EXPECT(cudaGetLastError() == cudaSuccess);

    const size_t n = 3 << 20;
    std::vector<uint32_t> h(n), back(n);
    for (size_t i = 0; i < n; ++i) h[i] = static_cast<uint32_t>(i * 2654435761u);

    uint32_t *a = nullptr, *b = nullptr;
    EXPECT(cudaMalloc(&a, n * 4) == cudaSuccess && a);
    EXPECT(cudaMalloc(&b, n * 4) == cudaSuccess && b);
    EXPECT(reinterpret_cast<uintptr_t>(a) % 256 == 0);
    EXPECT(cudaMemcpy(a, h.data(), n * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    EXPECT(cudaMemcpy(b, a, n * 4, cudaMemcpyDeviceToDevice) == cudaSuccess);
    EXPECT(cudaMemcpy(back.data(), b, n * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    EXPECT(back == h);

    // Offsets inside an allocation and cudaMemcpyDefault.
    EXPECT(cudaMemset(b + 10, 0xff, 4 * 5) == cudaSuccess);
    EXPECT(cudaMemcpy(back.data(), b, 64, cudaMemcpyDefault) == cudaSuccess);
    EXPECT(back[9] == h[9] && back[10] == 0xffffffffu && back[14] == 0xffffffffu && back[15] == h[15]);

    // Out of bounds, bad pointers, double free.
    EXPECT(cudaMemcpy(back.data(), a, n * 4 + 1, cudaMemcpyDeviceToHost) == cudaErrorInvalidValue);
    EXPECT(cudaMemset(h.data(), 0, 16) == cudaErrorInvalidValue);
    EXPECT(cudaFree(a + 1) == cudaErrorInvalidValue);
    EXPECT(cudaFree(nullptr) == cudaSuccess);

    size_t free_before = 0, total = 0, free_after = 0;
    EXPECT(cudaMemGetInfo(&free_before, &total) == cudaSuccess && free_before < total);
    EXPECT(cudaFree(a) == cudaSuccess);
    EXPECT(cudaFree(a) == cudaErrorInvalidValue);
    EXPECT(cudaFree(b) == cudaSuccess);
    EXPECT(cudaMemGetInfo(&free_after, &total) == cudaSuccess && free_after == total);

    void* huge = nullptr;
    EXPECT(cudaMalloc(&huge, total + 1) == cudaErrorMemoryAllocation && huge == nullptr);
    cudaGetLastError();
    // Managed memory (the shared pool): usable from the host directly.
    int* managed = nullptr;
    EXPECT(cudaMallocManaged(&managed, 64 * sizeof(int)) == cudaSuccess && managed != nullptr);
    if (managed) {
        for (int i = 0; i < 64; ++i) managed[i] = i;
        EXPECT(cudaMemset(managed, 0, 16 * sizeof(int)) == cudaSuccess && managed[0] == 0 && managed[63] == 63);
        EXPECT(cudaFree(managed) == cudaSuccess);
    }
    TEST_MAIN_END();
}
