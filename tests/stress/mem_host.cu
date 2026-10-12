// Page-locked host memory: cudaMallocHost, cudaHostAlloc (mapped, zero-copy
// through cudaHostGetDevicePointer), kernels reading cudaMallocHost memory
// directly through UVA, cudaHostRegister of a malloc'd buffer, and
// cudaPointerGetAttributes / cudaHostGetFlags.
#include "stress.cuh"

__global__ void square_into(const float* in, float* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) out[i] = in[i] * in[i];
}

__global__ void zero_copy_write(int* host_out, const int* host_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) host_out[i] = host_in[i] * 3 + i;
}

__global__ void add_one(unsigned* p, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) p[i] += 1;
}

int main() {
    using namespace stress;

    {  // cudaMallocHost round trip
        const size_t n = 8 << 20;  // floats, 32 MiB
        float *h_in, *h_out, *d;
        STRESS_CHECK(cudaMallocHost(&h_in, n * sizeof(float)));
        STRESS_CHECK(cudaMallocHost((void**)&h_out, n * sizeof(float)));
        STRESS_CHECK(cudaMalloc(&d, n * sizeof(float)));
        auto x = random_floats(n, 1);
        std::memcpy(h_in, x.data(), n * sizeof(float));
        double k = best_ms([&] {
            STRESS_CHECK(cudaMemcpy(d, h_in, n * sizeof(float), cudaMemcpyHostToDevice));
            STRESS_CHECK(cudaMemcpy(h_out, d, n * sizeof(float), cudaMemcpyDeviceToHost));
        });
        double ref = best_ms([&] {
#pragma omp parallel for
            for (size_t i = 0; i < n; ++i) h_out[i] = h_in[i];
        });
        std::memset(h_out, 0, n * sizeof(float));
        STRESS_CHECK(cudaMemcpy(d, h_in, n * sizeof(float), cudaMemcpyHostToDevice));
        STRESS_CHECK(cudaMemcpy(h_out, d, n * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = equal(h_out, x.data(), n);
        STRESS_CHECK(cudaFreeHost(h_in));
        STRESS_CHECK(cudaFreeHost(h_out));
        STRESS_CHECK(cudaFree(d));
        report("malloc_host_roundtrip", ok, k, ref);
    }

    {  // kernel reads and writes cudaMallocHost memory directly (UVA: the host pointer is valid on the device)
        const int n = 1 << 18;
        float *h_in, *h_out;
        STRESS_CHECK(cudaMallocHost(&h_in, n * sizeof(float)));
        STRESS_CHECK(cudaMallocHost(&h_out, n * sizeof(float)));
        auto x = random_floats(n, 2);
        std::memcpy(h_in, x.data(), n * sizeof(float));
        int uva = 0;
        STRESS_CHECK(cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, 0));
        double k = time_device([&] { square_into<<<128, 256>>>(h_in, h_out, n); });
        bool ok = uva == 1;
        for (int i = 0; i < n && ok; ++i) ok = h_out[i] == x[i] * x[i];
        STRESS_CHECK(cudaFreeHost(h_in));
        STRESS_CHECK(cudaFreeHost(h_out));
        report("malloc_host_uva_kernel", ok, k, -1);
    }

    {  // cudaHostAlloc(Mapped) + cudaHostGetDevicePointer: the kernel writes host memory
        const int n = 100000;
        int *h_in, *h_out, *d_in, *d_out;
        STRESS_CHECK(cudaHostAlloc(&h_in, n * sizeof(int), cudaHostAllocMapped));
        STRESS_CHECK(cudaHostAlloc((void**)&h_out, n * sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable));
        STRESS_CHECK(cudaHostGetDevicePointer(&d_in, h_in, 0));
        STRESS_CHECK(cudaHostGetDevicePointer((void**)&d_out, h_out, 0));
        for (int i = 0; i < n; ++i) h_in[i] = i * 7 - 300000, h_out[i] = -1;
        unsigned flags = 0;
        STRESS_CHECK(cudaHostGetFlags(&flags, h_out));
        double k = time_device([&] { zero_copy_write<<<(n + 255) / 256, 256>>>(d_out, d_in, n); });
        bool ok = (flags & cudaHostAllocMapped) && d_in && d_out;
        if (!ok) std::printf("  flags=%u\n", flags);
        for (int i = 0; i < n && ok; ++i) ok = h_out[i] == (i * 7 - 300000) * 3 + i;
        // The host updates the input; the next launch sees it without any copy.
        for (int i = 0; i < n; ++i) h_in[i] = i;
        zero_copy_write<<<(n + 255) / 256, 256>>>(d_out, d_in, n);
        STRESS_CHECK(cudaGetLastError());
        sync();
        for (int i = 0; i < n && ok; ++i) ok = h_out[i] == i * 4;
        STRESS_CHECK(cudaFreeHost(h_in));
        STRESS_CHECK(cudaFreeHost(h_out));
        report("host_alloc_mapped_zero_copy", ok, k, -1);
    }

    {  // cudaHostAlloc with default flags and write-combined, used for async copies
        const size_t n = 1 << 20;
        unsigned *a, *wc, *d;
        STRESS_CHECK(cudaHostAlloc(&a, n * 4, cudaHostAllocDefault));
        STRESS_CHECK(cudaHostAlloc(&wc, n * 4, cudaHostAllocWriteCombined));
        STRESS_CHECK(cudaMalloc(&d, n * 4));
        auto x = random_u32(n, 4);
        std::memcpy(wc, x.data(), n * 4);  // write-combined: the host only writes it
        cudaStream_t s;
        STRESS_CHECK(cudaStreamCreate(&s));
        double t0 = now_ms();
        STRESS_CHECK(cudaMemcpyAsync(d, wc, n * 4, cudaMemcpyHostToDevice, s));
        add_one<<<256, 256, 0, s>>>(d, n);
        STRESS_CHECK(cudaGetLastError());
        STRESS_CHECK(cudaMemcpyAsync(a, d, n * 4, cudaMemcpyDeviceToHost, s));
        STRESS_CHECK(cudaStreamSynchronize(s));
        double k = now_ms() - t0;
        bool ok = true;
        for (size_t i = 0; i < n && ok; ++i) ok = a[i] == x[i] + 1;
        STRESS_CHECK(cudaStreamDestroy(s));
        STRESS_CHECK(cudaFreeHost(a));
        STRESS_CHECK(cudaFreeHost(wc));
        STRESS_CHECK(cudaFree(d));
        report("host_alloc_default_writecombined", ok, k, -1);
    }

    {  // cudaHostRegister of a malloc'd buffer: async copies, then mapped access from a kernel
        const size_t n = 4 << 20;
        size_t bytes = n * sizeof(unsigned);
        unsigned* buf = static_cast<unsigned*>(std::malloc(bytes));
        auto x = random_u32(n, 6);
        std::memcpy(buf, x.data(), bytes);
        STRESS_CHECK(cudaHostRegister(buf, bytes, cudaHostRegisterDefault));
        unsigned* d;
        STRESS_CHECK(cudaMalloc(&d, bytes));
        cudaStream_t s;
        STRESS_CHECK(cudaStreamCreate(&s));
        double k = best_ms([&] {
            STRESS_CHECK(cudaMemcpyAsync(d, buf, bytes, cudaMemcpyHostToDevice, s));
            add_one<<<512, 256, 0, s>>>(d, n);
            STRESS_CHECK(cudaMemcpyAsync(buf, d, bytes, cudaMemcpyDeviceToHost, s));
            STRESS_CHECK(cudaStreamSynchronize(s));
        }, 2);
        // best_ms ran it 3 times.
        bool ok = true;
        for (size_t i = 0; i < n && ok; ++i) ok = buf[i] == x[i] + 3;
        STRESS_CHECK(cudaHostUnregister(buf));
        // Registered again as mapped: the kernel increments the host buffer in place.
        STRESS_CHECK(cudaHostRegister(buf, bytes, cudaHostRegisterMapped));
        unsigned* dev_view = nullptr;
        STRESS_CHECK(cudaHostGetDevicePointer(&dev_view, buf, 0));
        add_one<<<512, 256>>>(dev_view, n);
        STRESS_CHECK(cudaGetLastError());
        sync();
        for (size_t i = 0; i < n && ok; ++i) ok = buf[i] == x[i] + 4;
        STRESS_CHECK(cudaHostUnregister(buf));
        STRESS_CHECK(cudaStreamDestroy(s));
        STRESS_CHECK(cudaFree(d));
        std::free(buf);
        report("host_register_malloc", ok, k, -1);
    }

    {  // cudaPointerGetAttributes on every kind of pointer
        void *dev, *pinned, *managed;
        STRESS_CHECK(cudaMalloc(&dev, 4096));
        STRESS_CHECK(cudaMallocHost(&pinned, 4096));
        STRESS_CHECK(cudaMallocManaged(&managed, 4096));
        std::vector<char> pageable(4096);
        cudaPointerAttributes a{}, b{}, c{}, e{};
        STRESS_CHECK(cudaPointerGetAttributes(&a, dev));
        STRESS_CHECK(cudaPointerGetAttributes(&b, pinned));
        STRESS_CHECK(cudaPointerGetAttributes(&c, managed));
        STRESS_CHECK(cudaPointerGetAttributes(&e, pageable.data()));
        cudaPointerAttributes off{};
        STRESS_CHECK(cudaPointerGetAttributes(&off, static_cast<char*>(dev) + 100));
        bool ok = a.type == cudaMemoryTypeDevice && a.devicePointer == dev && a.device == 0 &&
                  b.type == cudaMemoryTypeHost && b.hostPointer == pinned && c.type == cudaMemoryTypeManaged &&
                  e.type == cudaMemoryTypeUnregistered && off.type == cudaMemoryTypeDevice &&
                  off.devicePointer == static_cast<char*>(dev) + 100;
        if (!ok) std::printf("  types %d %d %d %d %d\n", a.type, b.type, c.type, e.type, off.type);
        STRESS_CHECK(cudaFree(dev));
        STRESS_CHECK(cudaFreeHost(pinned));
        STRESS_CHECK(cudaFree(managed));
        report("pointer_attributes", ok, 0, -1);
    }

    return finish();
}
