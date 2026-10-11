#include "device/properties.h"

#include <cstring>
#include <fstream>
#include <functional>
#include <string>

#include "common/config.h"
#include "common/topology.h"

namespace cpusim {

namespace {

// Stable per machine and device, so cpu-smi and cudaGetDeviceProperties agree across runs.
void make_uuid(const std::string& seed, uint8_t out[16]) {
    std::string machine;
    std::ifstream("/etc/machine-id") >> machine;
    uint64_t h1 = std::hash<std::string>{}(machine + "/" + seed);
    uint64_t h2 = std::hash<std::string>{}(seed + "/" + machine + "/cpusim");
    std::memcpy(out, &h1, 8);
    std::memcpy(out + 8, &h2, 8);
    out[6] = (out[6] & 0x0f) | 0x50;  // RFC 4122 layout, version 5 style
    out[8] = (out[8] & 0x3f) | 0x80;
}

}  // namespace

DeviceProperties make_properties(const Config& cfg, const Topology& topo, int cores, uint64_t vram,
                                 int max_clock_khz) {
    DeviceProperties p{};
    std::string name = cfg.name.empty() ? short_cpu_model(topo.cpu_model) : cfg.name;
    std::strncpy(p.name, name.c_str(), sizeof(p.name) - 1);
    make_uuid(topo.cpu_model + "/0", p.uuid);

    p.totalGlobalMem = vram;
    p.sharedMemPerBlock = 48 * 1024;
    p.regsPerBlock = 65536;
    p.warpSize = 32;
    p.memPitch = 2147483647;
    p.maxThreadsPerBlock = 1024;
    p.maxThreadsDim[0] = 1024;
    p.maxThreadsDim[1] = 1024;
    p.maxThreadsDim[2] = 64;
    p.maxGridSize[0] = 2147483647;
    p.maxGridSize[1] = 65535;
    p.maxGridSize[2] = 65535;
    p.totalConstMem = 64 * 1024;
    // From the dot product instructions the tensor core emulation can use.
    const int cc = compute_capability(tensor_level());
    p.major = cc / 10;
    p.minor = cc % 10;
    p.textureAlignment = 512;
    p.texturePitchAlignment = 32;
    p.multiProcessorCount = cores;
    p.concurrentKernels = 1;
    p.asyncEngineCount = 2;
    p.unifiedAddressing = 1;
    p.memoryBusWidth = 128;
    p.l2CacheSize = 32 * 1024 * 1024;
    p.persistingL2CacheMaxSize = 0;
    p.maxThreadsPerMultiProcessor = 1536;
    p.streamPrioritiesSupported = 1;
    p.globalL1CacheSupported = 1;
    p.localL1CacheSupported = 1;
    p.sharedMemPerMultiprocessor = 100 * 1024;
    p.regsPerMultiprocessor = 65536;
    p.computePreemptionSupported = 0;
    p.cooperativeLaunch = 0;
    p.sharedMemPerBlockOptin = 99 * 1024;
    p.maxBlocksPerMultiProcessor = 24;
    p.reservedSharedMemPerBlock = 1024;
    p.clockRate = max_clock_khz;
    p.memoryClockRate = 0;

    // Discrete GPU without host mapping while there is no shared pool (DESIGN.md 6.6).
    bool shared = cfg.vram_shared > 0;
    p.integrated = shared;
    p.canMapHostMemory = shared;
    p.managedMemory = shared;
    p.concurrentManagedAccess = shared;
    p.pageableMemoryAccess = 0;
    p.hostRegisterSupported = 0;
    p.memoryPoolsSupported = 0;
    return p;
}

}  // namespace cpusim
