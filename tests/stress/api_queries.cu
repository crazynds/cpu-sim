// Device and runtime queries: device count and properties, attributes,
// occupancy calculators, memory info, versions, cudaSetDevice, and
// cudaDeviceReset followed by fresh allocations and launches.
#include "stress.cuh"

__global__ void small_kernel(float* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = float(i) * 2.0f;
}

__global__ void heavy_smem_kernel(float* p) {
    __shared__ float s[8192];  // 32 KiB static
    s[threadIdx.x] = float(threadIdx.x);
    __syncthreads();
    p[blockIdx.x * blockDim.x + threadIdx.x] = s[(threadIdx.x + 1) % blockDim.x];
}

int main() {
    using namespace stress;

    {  // device count, cudaSetDevice, cudaGetDevice
        int count = -1, dev = -1;
        STRESS_CHECK(cudaGetDeviceCount(&count));
        STRESS_CHECK(cudaSetDevice(0));
        STRESS_CHECK(cudaGetDevice(&dev));
        report("device_count_set_get", count >= 1 && dev == 0, 0, -1);
    }

    {  // cudaGetDeviceProperties: sane values
        cudaDeviceProp p{};
        STRESS_CHECK(cudaGetDeviceProperties(&p, 0));
        bool ok = p.warpSize == 32 && p.maxThreadsPerBlock >= 1024 && p.sharedMemPerBlock >= 48 * 1024 &&
                  p.multiProcessorCount > 0 && p.maxThreadsDim[0] >= 1024 && p.maxThreadsDim[1] >= 1024 &&
                  p.maxThreadsDim[2] >= 64 && p.maxGridSize[0] >= 2147483647 && p.maxGridSize[1] >= 65535 &&
                  p.maxGridSize[2] >= 65535 && p.totalGlobalMem > (size_t(1) << 30) && p.major >= 3 &&
                  p.regsPerBlock > 0 && p.maxThreadsPerMultiProcessor >= p.maxThreadsPerBlock &&
                  p.sharedMemPerBlockOptin >= p.sharedMemPerBlock && p.totalConstMem >= 64 * 1024 &&
                  p.unifiedAddressing == 1 && std::strlen(p.name) > 0;
        std::printf("  %s: cc %d.%d, %d SMs, %zu MiB, smem/block %zu (opt-in %zu)\n", p.name, p.major, p.minor,
                    p.multiProcessorCount, p.totalGlobalMem >> 20, p.sharedMemPerBlock, p.sharedMemPerBlockOptin);
        report("device_properties", ok, 0, -1);
    }

    {  // cudaDeviceGetAttribute agrees with the properties
        cudaDeviceProp p{};
        STRESS_CHECK(cudaGetDeviceProperties(&p, 0));
        int warp, mtpb, sms, smem, major, minor, gx, l2, clock;
        STRESS_CHECK(cudaDeviceGetAttribute(&warp, cudaDevAttrWarpSize, 0));
        STRESS_CHECK(cudaDeviceGetAttribute(&mtpb, cudaDevAttrMaxThreadsPerBlock, 0));
        STRESS_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
        STRESS_CHECK(cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlock, 0));
        STRESS_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
        STRESS_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0));
        STRESS_CHECK(cudaDeviceGetAttribute(&gx, cudaDevAttrMaxGridDimX, 0));
        STRESS_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
        STRESS_CHECK(cudaDeviceGetAttribute(&clock, cudaDevAttrClockRate, 0));
        bool ok = warp == p.warpSize && mtpb == p.maxThreadsPerBlock && sms == p.multiProcessorCount &&
                  size_t(smem) == p.sharedMemPerBlock && major == p.major && minor == p.minor && gx == p.maxGridSize[0] &&
                  l2 == p.l2CacheSize && clock > 0;
        report("device_get_attribute", ok, 0, -1);
    }

    {  // occupancy: blocks per SM and the suggested block size
        int nb256 = 0, nb1024 = 0, nb_smem = 0, min_grid = 0, block = 0;
        STRESS_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb256, small_kernel, 256, 0));
        STRESS_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb1024, small_kernel, 1024, 0));
        STRESS_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb_smem, heavy_smem_kernel, 128, 0));
        STRESS_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, small_kernel, 0, 0));
        cudaDeviceProp p{};
        STRESS_CHECK(cudaGetDeviceProperties(&p, 0));
        size_t smem_per_sm = p.sharedMemPerMultiprocessor;
        bool ok = nb256 >= 1 && nb1024 >= 1 && nb256 >= nb1024 && nb256 * 256 <= p.maxThreadsPerMultiProcessor &&
                  nb_smem >= 1 && size_t(nb_smem) * 32768 <= smem_per_sm && block > 0 && block <= 1024 &&
                  block % 32 == 0 && min_grid >= p.multiProcessorCount;
        std::printf("  blocks/SM: 256->%d 1024->%d smem32K->%d; suggested block %d, min grid %d\n", nb256, nb1024, nb_smem,
                    block, min_grid);
        // The suggestion must be usable.
        const int n = 100000;
        Buffer<float> out(n);
        double k = time_device([&] { small_kernel<<<(n + block - 1) / block, block>>>(out, n); });
        auto got = out.download();
        for (int i = 0; i < n && ok; ++i) ok = got[i] == float(i) * 2.0f;
        report("occupancy_queries", ok, k, -1);
    }

    {  // cudaMemGetInfo tracks an allocation
        size_t free0 = 0, total = 0, free1 = 0, free2 = 0;
        STRESS_CHECK(cudaMemGetInfo(&free0, &total));
        void* p;
        const size_t big = 256 << 20;
        STRESS_CHECK(cudaMalloc(&p, big));
        STRESS_CHECK(cudaMemGetInfo(&free1, &total));
        STRESS_CHECK(cudaFree(p));
        STRESS_CHECK(cudaMemGetInfo(&free2, &total));
        bool ok = total > (size_t(1) << 30) && free0 <= total && free1 + big / 2 <= free0 && free2 >= free1 + big / 2;
        if (!ok) std::printf("  free %zu -> %zu -> %zu of %zu\n", free0, free1, free2, total);
        report("mem_get_info", ok, 0, -1);
    }

    {  // runtime and driver versions
        int rt = 0, drv = 0;
        STRESS_CHECK(cudaRuntimeGetVersion(&rt));
        STRESS_CHECK(cudaDriverGetVersion(&drv));
        std::printf("  runtime %d driver %d (CUDART_VERSION %d)\n", rt, drv, CUDART_VERSION);
        report("versions", rt >= 11000 && drv >= rt && rt == CUDART_VERSION, 0, -1);
    }

    {  // device flags, limits, cache config: setters accepted, getters consistent
        size_t stack = 0, fifo = 0;
        STRESS_CHECK(cudaDeviceGetLimit(&stack, cudaLimitStackSize));
        STRESS_CHECK(cudaDeviceGetLimit(&fifo, cudaLimitPrintfFifoSize));
        STRESS_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));
        size_t stack2 = 0;
        STRESS_CHECK(cudaDeviceGetLimit(&stack2, cudaLimitStackSize));
        STRESS_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
        cudaFuncCache cc;
        STRESS_CHECK(cudaDeviceGetCacheConfig(&cc));
        STRESS_CHECK(cudaFuncSetCacheConfig(small_kernel, cudaFuncCachePreferL1));
        unsigned flags = 0;
        STRESS_CHECK(cudaGetDeviceFlags(&flags));
        bool ok = stack > 0 && fifo > 0 && stack2 >= 4096;
        report("limits_cache_flags", ok, 0, -1);
    }

    {  // cudaDeviceReset: everything is gone, new allocations and launches work
        float* before;
        STRESS_CHECK(cudaMalloc(&before, 1 << 20));
        STRESS_CHECK(cudaDeviceReset());
        STRESS_CHECK(cudaSetDevice(0));
        const int n = 1 << 18;
        double t0 = now_ms();
        float* p;
        STRESS_CHECK(cudaMalloc(&p, n * sizeof(float)));
        small_kernel<<<n / 256, 256>>>(p, n);
        STRESS_CHECK(cudaGetLastError());
        sync();
        std::vector<float> got(n);
        STRESS_CHECK(cudaMemcpy(got.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
        double k = now_ms() - t0;
        bool ok = true;
        for (int i = 0; i < n && ok; ++i) ok = got[i] == float(i) * 2.0f;
        // A second reset with live streams and events.
        cudaStream_t s;
        cudaEvent_t e;
        STRESS_CHECK(cudaStreamCreate(&s));
        STRESS_CHECK(cudaEventCreate(&e));
        STRESS_CHECK(cudaDeviceReset());
        Buffer<float> q(1000);
        small_kernel<<<4, 256>>>(q, 1000);
        STRESS_CHECK(cudaGetLastError());
        auto g2 = q.download();
        for (int i = 0; i < 1000 && ok; ++i) ok = g2[i] == float(i) * 2.0f;
        report("device_reset_then_use", ok, k, -1);
    }

    return finish();
}
