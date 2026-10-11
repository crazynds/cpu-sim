#pragma once

#include <cstdint>

namespace cpusim {

struct Config;
struct Topology;

// Plain data, sent as-is over the driver socket. Field names follow cudaDeviceProp;
// the runtime copies them into its own struct. Defaults come from the reference
// GPU (RTX 5060 Laptop, compute capability 12.0) except where the CPU decides.
struct DeviceProperties {
    char name[256];
    uint8_t uuid[16];
    uint64_t totalGlobalMem;
    uint64_t sharedMemPerBlock;
    int32_t regsPerBlock;
    int32_t warpSize;
    uint64_t memPitch;
    int32_t maxThreadsPerBlock;
    int32_t maxThreadsDim[3];
    int32_t maxGridSize[3];
    uint64_t totalConstMem;
    int32_t major;
    int32_t minor;
    uint64_t textureAlignment;
    uint64_t texturePitchAlignment;
    int32_t multiProcessorCount;
    int32_t integrated;
    int32_t canMapHostMemory;
    int32_t concurrentKernels;
    int32_t ECCEnabled;
    int32_t pciBusID;
    int32_t pciDeviceID;
    int32_t pciDomainID;
    int32_t asyncEngineCount;
    int32_t unifiedAddressing;
    int32_t memoryBusWidth;
    int32_t l2CacheSize;
    int32_t persistingL2CacheMaxSize;
    int32_t maxThreadsPerMultiProcessor;
    int32_t streamPrioritiesSupported;
    int32_t globalL1CacheSupported;
    int32_t localL1CacheSupported;
    uint64_t sharedMemPerMultiprocessor;
    int32_t regsPerMultiprocessor;
    int32_t managedMemory;
    int32_t isMultiGpuBoard;
    int32_t pageableMemoryAccess;
    int32_t concurrentManagedAccess;
    int32_t computePreemptionSupported;
    int32_t cooperativeLaunch;
    uint64_t sharedMemPerBlockOptin;
    int32_t maxBlocksPerMultiProcessor;
    uint64_t reservedSharedMemPerBlock;
    int32_t hostRegisterSupported;
    int32_t memoryPoolsSupported;
    int32_t clockRate;        // kHz, the reserved cores' max frequency
    int32_t memoryClockRate;  // kHz
};

// Fills the properties for a device with the given reserved cores and VRAM.
DeviceProperties make_properties(const Config& cfg, const Topology& topo, int cores, uint64_t vram,
                                 int max_clock_khz);

}  // namespace cpusim
