// Managed and mapped host memory: written by the host, read and written by
// kernels, read back by the host without copies; copies and memsets between
// managed memory, VRAM and host memory; the CUDA 12 and 13 prefetch forms.
#include <vector>

#include "check.cuh"

__global__ void scale(float* a, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = a[i] * s + float(i);
}

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    Checker c("managed");
    const int n = 1 << 16;
    cudaDeviceProp p;
    CHECK_CUDA(cudaGetDeviceProperties(&p, 0));
    c.expect(p.managedMemory == 1, "managedMemory property", -1, p.managedMemory, 1);

    float* m;
    CHECK_CUDA(cudaMallocManaged(&m, n * sizeof(float)));
    for (int i = 0; i < n; ++i) m[i] = 1.0f;
    CHECK_CUDA(cudaMemPrefetchAsync(m, n * sizeof(float), 0));                                   // CUDA 12 form
    CHECK_CUDA(cudaMemPrefetchAsync(m, n * sizeof(float), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0));  // 13
    CHECK_CUDA(cudaMemAdvise(m, n * sizeof(float), cudaMemAdviseSetReadMostly, 0));
    scale<<<n / 256, 256>>>(m, n, 2.0f);
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int i = 0; i < n; ++i) c.expect(m[i] == 2.0f + float(i), "kernel result read by the host", i, m[i], 2.0f + i);

    float* d;
    CHECK_CUDA(cudaMalloc(&d, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d, m, n * sizeof(float), cudaMemcpyDefault));         // managed -> VRAM
    CHECK_CUDA(cudaMemset(m, 0, n * sizeof(float)));                           // memset on managed
    for (int i = 0; i < n; ++i) c.expect(m[i] == 0.0f, "memset", i, m[i], 0);

    float* h;
    CHECK_CUDA(cudaHostAlloc(&h, n * sizeof(float), cudaHostAllocMapped));
    float* hd;
    CHECK_CUDA(cudaHostGetDevicePointer(&hd, h, 0));
    for (int i = 0; i < n; ++i) h[i] = 0.5f;
    add<<<n / 256, 256>>>(d, hd, m, n);                                        // VRAM + mapped host -> managed
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int i = 0; i < n; ++i) c.expect(m[i] == 2.5f + float(i), "VRAM + mapped -> managed", i, m[i], 2.5f + i);

    std::vector<float> back(n);
    CHECK_CUDA(cudaMemcpy(back.data(), m, n * sizeof(float), cudaMemcpyDeviceToHost));  // managed -> host
    c.expect(back[100] == 102.5f, "managed -> host copy", 100, back[100], 102.5f);
    CHECK_CUDA(cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost));                  // VRAM -> mapped host
    c.expect(h[3] == 5.0f, "VRAM -> mapped host", 3, h[3], 5.0f);

    CHECK_CUDA(cudaFree(m));
    CHECK_CUDA(cudaFree(d));
    CHECK_CUDA(cudaFreeHost(h));
    return c.finish();
}
