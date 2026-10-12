#pragma once

// Wire format between the application (daemon backend) and the driver, over a
// SOCK_SEQPACKET Unix socket: one request, one reply, each a single packet.
// Same machine and same build, so structs travel as raw bytes; the version in
// Hello rejects mismatched builds.

#include <cstdint>

#include "device/kernel_abi.h"
#include "device/properties.h"

namespace cpusim::proto {

constexpr uint32_t kMagic = 0x43505553;  // "CPUS"
constexpr uint32_t kVersion = 6;
constexpr int32_t kCudaVersion = 13000;   // the CUDA version we implement

enum class Op : uint32_t {
    Hello = 1,     // a = kVersion, b = flags (kHelloMonitor)
    Malloc,        // a = bytes                   -> reply.a = device pointer
    Free,          // a = device pointer
    MemcpyHtoD,    // a = device dst, b = host src (client address), c = bytes
    MemcpyDtoH,    // a = host dst (client address), b = device src, c = bytes
    MemcpyDtoD,    // a = device dst, b = device src, c = bytes
    Memset,        // a = device dst, b = value, c = bytes
    MemGetInfo,    //                             -> reply.a = free, reply.b = total
    Synchronize,
    Status,        //                             -> DeviceStatus
    LoadModule,    // fd of the shared object (SCM_RIGHTS)  -> reply.a = module
    GetFunction,   // a = module, payload = kernel name     -> reply.a = function
    Launch,        // a = function, payload = LaunchPayload + parameter bytes
    SharedPool,    // -> fd of the shared pool (SCM_RIGHTS), reply.a = its address, reply.b = size; error if none
    MallocManaged, // a = bytes                   -> reply.a = pointer in the shared pool
    BindSymbols,   // a = module, b = 1: copy the initial values, payload = Device::bind_symbols table
};

// Requests may carry a payload right after the Request header, in the same packet.
constexpr size_t kMaxPayload = 64 * 1024;

struct LaunchPayload {
    uint32_t grid[3];
    uint32_t block[3];
    uint32_t shared_bytes;
    uint32_t param_bytes;
    uint32_t profile;
    uint32_t debug;
};


// Monitoring tools (cpu-smi) connect without creating a context.
constexpr uint64_t kHelloMonitor = 1;
// More connections of the same process join its context (Hello's c = the
// context id): one per stream in flight, so streams do not wait for each
// other on the socket. Closing one of them leaves the context open.
constexpr uint64_t kHelloAttach = 2;

struct Request {
    uint32_t magic;
    Op op;
    uint64_t a, b, c;
};

struct Reply {
    int32_t error;  // cpusim::Error
    uint32_t reserved;
    uint64_t a, b;
};

struct LaunchReply {
    Reply header;
    uint64_t kernel_ns;
    uint64_t counters[CPUSIM_COUNTERS];
    char report[1024];
};

struct HelloReply {
    Reply header;
    int32_t driver_pid;
    uint32_t context;
    uint64_t vram_base;
    uint64_t vram_size;
    char driver_version[32];
    DeviceProperties props;
};

constexpr int kMaxProcesses = 64;
constexpr int kMaxFans = 4;

struct ProcessStatus {
    int32_t pid;
    uint32_t context;
    int32_t inprocess;  // runs the device in its own process (CPUSIM_BACKEND=inprocess)
    int32_t reserved;
    uint64_t used_bytes;
    uint64_t allocations;
    int64_t started_unix;
    char name[256];
};

// Everything cpu-smi shows. Negative floating point values mean N/A.
struct DeviceStatus {
    Reply header;
    char driver_version[32];
    int32_t cuda_version;
    int32_t driver_pid;
    int64_t driver_started_unix;
    char config_source[256];

    DeviceProperties props;
    char cpu_model[128];
    char reserved_cpus[256];      // logical CPU list, "4-15,20-31"
    int32_t reserved_cores;
    int32_t host_cores;
    int32_t host_logical_cpus;

    uint64_t vram_total, vram_used, vram_free, vram_largest_free, vram_resident;
    uint64_t vram_shared_total, vram_shared_used;
    uint64_t vram_base;
    uint64_t page_size;
    char hugepages[16];
    int32_t prefault;

    double gpu_util;              // % of the last sample window with a kernel running
    double memory_util;           // % of the window with a copy or memset running
    double host_load;             // % busy of the reserved CPUs, any process
    double temperature_c;
    double power_w;
    double clock_mhz;
    double max_clock_mhz;
    int32_t fan_count;
    int32_t fan_rpm[kMaxFans];
    int32_t pstate;               // 0 busy, 2 copying, 8 idle

    uint64_t kernels_launched, kernels_profiled;
    uint64_t bytes_htod, bytes_dtoh, bytes_dtod, bytes_memset;
    uint64_t alloc_calls, free_calls, failed_allocs;

    int32_t process_count;
    ProcessStatus processes[kMaxProcesses];
};

}  // namespace cpusim::proto
