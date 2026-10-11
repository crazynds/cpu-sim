// deviceQuery: prints what cudaGetDeviceProperties reports for device 0.
#include <cstdio>

#include "cuda_runtime.h"

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        std::printf("cudaGetDeviceCount: %s (%s)\n", cudaGetErrorName(err), cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);

    std::printf("Device 0: \"%s\"\n", p.name);
    std::printf("  CUDA Driver / Runtime Version          %d.%d / %d.%d\n", drv / 1000, drv % 100 / 10, rt / 1000,
                rt % 100 / 10);
    std::printf("  Compute Capability                     %d.%d\n", p.major, p.minor);
    std::printf("  Global memory                          %zu MiB (%zu MiB free)\n", p.totalGlobalMem >> 20, free >> 20);
    std::printf("  Multiprocessors (physical cores)       %d\n", p.multiProcessorCount);
    std::printf("  Clock rate                             %.0f MHz\n", p.clockRate / 1000.0);
    std::printf("  L2 cache                               %d KiB\n", p.l2CacheSize / 1024);
    std::printf("  Constant memory                        %zu bytes\n", p.totalConstMem);
    std::printf("  Shared memory per block / per SM       %zu / %zu bytes\n", p.sharedMemPerBlock,
                p.sharedMemPerMultiprocessor);
    std::printf("  Registers per block                    %d\n", p.regsPerBlock);
    std::printf("  Warp size                              %d\n", p.warpSize);
    std::printf("  Max threads per SM / per block         %d / %d\n", p.maxThreadsPerMultiProcessor,
                p.maxThreadsPerBlock);
    std::printf("  Max block dimensions                   (%d, %d, %d)\n", p.maxThreadsDim[0], p.maxThreadsDim[1],
                p.maxThreadsDim[2]);
    std::printf("  Max grid dimensions                    (%d, %d, %d)\n", p.maxGridSize[0], p.maxGridSize[1],
                p.maxGridSize[2]);
    std::printf("  Integrated / managed memory            %s / %s\n", p.integrated ? "Yes" : "No",
                p.managedMemory ? "Yes" : "No");
    std::printf("  Unified addressing                     %s\n", p.unifiedAddressing ? "Yes" : "No");
    return 0;
}
