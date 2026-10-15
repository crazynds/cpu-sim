// The CUDA runtime: kernel launches, kernel attributes and occupancy, and
// the launch profiling and timing (CPUSIM_PROFILE, CPUSIM_KERNEL_TIMES).
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

#include <fcntl.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

extern char** environ;

#include "common/cpu_id.h"
#include "cudart/streams.h"
#include "cuda_runtime_api.h"
#include "device/device.h"
#include "device/kernel_abi.h"
#include "cudart/runtime_internal.h"

using namespace cpusim::rt;

#pragma GCC visibility push(hidden)
namespace cpusim::rt {

// ---- profiling (CPUSIM_PROFILE=1)
//
// The first launch of each (kernel, grid, block) runs the instrumented entry,
// which counts operations; later launches of the same shape run the fast one
// and are timed. Rates combine both: counts from the instrumented run, time
// from the uninstrumented ones. Each launch still runs exactly once, so kernels
// that accumulate into their outputs keep working. Counts of data-dependent
// kernels are those of the first launch.

// CPUSIM_KERNEL_TIMES=1: each kernel's launches and time on the device,
// printed when the program ends ("cpusim-times <kernel> <launches> <ns>",
// and a total). For tracking performance across compiler changes with one
// measure for every program (bench/track).
struct KernelTimes {
    bool enabled = false;
    std::mutex mutex;
    std::map<std::string, std::pair<uint64_t, uint64_t>> by_kernel;  // launches, ns

    KernelTimes() {
        const char* v = std::getenv("CPUSIM_KERNEL_TIMES");
        enabled = v && *v && std::strcmp(v, "0") != 0;
        if (enabled) std::atexit([] { report(); });
    }
    void add(const std::string& kernel, uint64_t ns) {
        std::lock_guard lock(mutex);
        auto& e = by_kernel[kernel];
        e.first += 1;
        e.second += ns;
    }
    static void report();
};
KernelTimes& kernel_times() {
    static KernelTimes* t = new KernelTimes;  // never destroyed: used from atexit
    return *t;
}
void KernelTimes::report() {
    KernelTimes& t = kernel_times();
    std::lock_guard lock(t.mutex);
    uint64_t total = 0;
    for (const auto& [k, e] : t.by_kernel) {
        std::fprintf(stderr, "cpusim-times %s %llu %llu\n", k.c_str(), (unsigned long long)e.first,
                     (unsigned long long)e.second);
        total += e.second;
    }
    std::fprintf(stderr, "cpusim-times-total %llu\n", (unsigned long long)total);
}

struct ProfileEntry {
    std::string kernel;
    uint32_t grid[3], block[3];
    uint64_t counters[CPUSIM_COUNTERS] = {};
    uint64_t profiled_ns = 0;
    uint64_t timed_runs = 0, timed_ns = 0;
};

struct Profiler {
    bool enabled = false;
    std::mutex mutex;
    std::vector<ProfileEntry> entries;

    Profiler() {
        const char* v = std::getenv("CPUSIM_PROFILE");
        enabled = v && *v && std::strcmp(v, "0") != 0;
        if (enabled) std::atexit([] { report(); });
    }

    ProfileEntry* find(const std::string& kernel, const cpusim::LaunchConfig& cfg) {
        for (auto& e : entries)
            if (e.kernel == kernel && !std::memcmp(e.grid, cfg.grid, sizeof(e.grid)) &&
                !std::memcmp(e.block, cfg.block, sizeof(e.block)))
                return &e;
        return nullptr;
    }

    static void report();
};

// Never destroyed: the report runs from atexit, after static destructors may have run.
Profiler& profiler() {
    static Profiler* p = new Profiler;
    return *p;
}

std::string human(double v, const char* unit) {
    const char* prefixes[] = {"", "K", "M", "G", "T", "P"};
    int i = 0;
    while (v >= 1000 && i < 5) v /= 1000, ++i;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3g %s%s", v, prefixes[i], unit);
    return buf;
}

void Profiler::report() {
    Profiler& p = profiler();
    std::lock_guard lock(p.mutex);
    std::fflush(stdout);
    std::fprintf(stderr, "\n==== cpusim profile (%zu kernel shapes) ====\n", p.entries.size());
    for (const auto& e : p.entries) {
        const uint64_t* c = e.counters;
        double secs = e.timed_runs ? e.timed_ns / 1e9 / e.timed_runs : 0;
        double flops = double(c[CPUSIM_CNT_FP32_FLOPS] + c[CPUSIM_CNT_FP64_FLOPS]);
        double bytes = double(c[CPUSIM_CNT_LOAD_BYTES] + c[CPUSIM_CNT_STORE_BYTES]);
        double lanes = c[CPUSIM_CNT_BLOCK_RUNS] ? double(c[CPUSIM_CNT_ACTIVE_LANES]) / c[CPUSIM_CNT_BLOCK_RUNS] : 0;
        std::fprintf(stderr, "%s  grid (%u,%u,%u) block (%u,%u,%u)\n", e.kernel.c_str(), e.grid[0], e.grid[1],
                     e.grid[2], e.block[0], e.block[1], e.block[2]);
        if (e.timed_runs)
            std::fprintf(stderr, "  time            %.3f ms (mean of %llu uninstrumented launches)\n", secs * 1e3,
                         (unsigned long long)e.timed_runs);
        else
            std::fprintf(stderr, "  time            only the instrumented launch ran (%.3f ms, not representative)\n",
                         e.profiled_ns / 1e6);
        std::fprintf(stderr, "  instrumented    %.3f ms (first launch, counting)\n", e.profiled_ns / 1e6);
        std::fprintf(stderr, "  fp32 flops      %s (%llu)\n", human(double(c[CPUSIM_CNT_FP32_FLOPS]), "").c_str(),
                     (unsigned long long)c[CPUSIM_CNT_FP32_FLOPS]);
        std::fprintf(stderr, "  fp64 flops      %s (%llu)\n", human(double(c[CPUSIM_CNT_FP64_FLOPS]), "").c_str(),
                     (unsigned long long)c[CPUSIM_CNT_FP64_FLOPS]);
        std::fprintf(stderr, "  special fp ops  %s\n", human(double(c[CPUSIM_CNT_FP_SPECIAL]), "").c_str());
        std::fprintf(stderr, "  integer ops     %s (%llu)\n", human(double(c[CPUSIM_CNT_INT_OPS]), "").c_str(),
                     (unsigned long long)c[CPUSIM_CNT_INT_OPS]);
        std::fprintf(stderr, "  loads / stores  %s / %s (%s / %s)\n", human(double(c[CPUSIM_CNT_LOADS]), "").c_str(),
                     human(double(c[CPUSIM_CNT_STORES]), "").c_str(), human(double(c[CPUSIM_CNT_LOAD_BYTES]), "B").c_str(),
                     human(double(c[CPUSIM_CNT_STORE_BYTES]), "B").c_str());
        std::fprintf(stderr, "  branches        %s\n", human(double(c[CPUSIM_CNT_BRANCHES]), "").c_str());
        std::fprintf(stderr, "  SIMD efficiency %.1f%% (%.1f of %d lanes active per block executed)\n",
                     100.0 * lanes / warp_size(), lanes, warp_size());
        if (secs > 0) {
            std::fprintf(stderr, "  throughput      %s, %s, %s\n", human(flops / secs, "FLOP/s").c_str(),
                         human(double(c[CPUSIM_CNT_INT_OPS]) / secs, "IOP/s").c_str(),
                         human(bytes / secs, "B/s").c_str());
            std::fprintf(stderr, "  intensity       %.2f FLOP/B\n", bytes ? flops / bytes : 0.0);
        }
    }
}


}  // namespace cpusim::rt
#pragma GCC visibility pop

extern "C" {

cudaError_t cudaLaunchKernel(const void* func, dim3 grid, dim3 block, void** args, size_t shared,
                             cudaStream_t stream) {
    REQUIRE_DEVICE(d);
    if (CUgraph_st* g = capturing(stream)) return set_kernel(add_node(g, CUgraphNode_st::Kernel), func, grid, block, shared, args);
    {
        // A bad configuration is reported by the launch itself, as in CUDA.
        const auto& p = d->properties();
        const uint32_t g[3] = {grid.x, grid.y, grid.z}, b[3] = {block.x, block.y, block.z};
        uint64_t threads = 1;
        for (int i = 0; i < 3; ++i) {
            if (!g[i] || !b[i] || b[i] > uint32_t(p.maxThreadsDim[i]) || g[i] > uint32_t(p.maxGridSize[i]))
                return record(Error::InvalidConfiguration);
            threads *= b[i];
        }
        if (threads > uint64_t(p.maxThreadsPerBlock)) return record(Error::InvalidConfiguration);
        if (shared > p.sharedMemPerBlockOptin) return record(Error::InvalidValue);
    }
    Registry& r = registry();
    std::unique_lock lock(r.mutex);
    auto it = r.functions.find(func);
    if (it == r.functions.end() || !it->second.module->header) return record(Error::InvalidDeviceFunction);
    Function& f = it->second;
    if (!f.kernel) {
        // Registered, but the compiler did not produce code for it.
        return record(Error::NoKernelImageForDevice);
    }
    // Tensor core types the CPU cannot run natively (no AVX512-BF16 for bf16,
    // no AVX512-VNNI for int8): like a binary built for a newer GPU.
    if (f.kernel->min_sm > d->properties().major * 10 + d->properties().minor)
        return record(Error::NoKernelImageForDevice);
    Module& m = *f.module;
    if (Error e = load_generic(m, d); e != Error::Success) return record(e);
    if (!f.handle) {
        Error e = d->get_function(m.handle, f.kernel->name, &f.handle);
        if (e != Error::Success) return record(e);
    }
    const KernelDesc& k = *f.kernel;
    uint64_t function = f.handle;
    // Instrumented launches run the generic module (the JIT one has no
    // instrumented entries).
    if (m.foreign) {
        static std::once_flag told;
        std::call_once(told, [] {
            std::fprintf(stderr, "cpusim: the kernels were compiled for another kind of CPU (cucc --mcpu=native "
                                 "on another machine); rebuild them here, or with --mcpu=<cpu> for both\n");
        });
        return record(Error::NoKernelImageForDevice);
    }
    bool specialized = false;
    // Fewer warps than cores, lanes that never look at each other: the
    // generic entry, whose lanes the executor can spread over the cores
    // (randomAccess: one block of 128 threads). The specialized one has its
    // masks built in.
    const uint64_t warps = (uint64_t(block.x) * block.y * block.z + 31) / 32;
    const bool spread = k.lane_independent && shared == 0 &&
                        uint64_t(grid.x) * grid.y * grid.z * warps < d->properties().multiProcessorCount;
    if (!profiler().enabled && !spread)
        if (uint64_t jf = jit_function(m, k, {block.x, block.y, block.z}, d)) function = jf, specialized = true;
    if (dump_asm_enabled()) dump_asm(m, specialized ? std::array<uint32_t, 3>{block.x, block.y, block.z}
                                                    : std::array<uint32_t, 3>{0, 0, 0});
    lock.unlock();

    std::vector<uint8_t> params(k.param_bytes);
    for (size_t i = 0; i < k.params.size(); ++i) std::memcpy(params.data() + k.params[i].first, args[i], k.params[i].second);

    cpusim::LaunchConfig cfg;
    cfg.grid[0] = grid.x, cfg.grid[1] = grid.y, cfg.grid[2] = grid.z;
    cfg.block[0] = block.x, cfg.block[1] = block.y, cfg.block[2] = block.z;
    cfg.shared_bytes = static_cast<uint32_t>(shared);

    // CPUSIM_DEBUG=1: the driver runs the launch step by step with guard pages
    // and reports the first invalid access; =2 also traces every step.
    static const int debug = [] {
        const char* v = std::getenv("CPUSIM_DEBUG");
        return v ? std::atoi(v) : 0;
    }();
    cfg.debug = debug;

    Profiler& prof = profiler();
    // __managed__ variables of the kernel's module: the host's values in,
    // the kernel's out (see the device header).
    struct Managed {
        uint64_t address;
        void* host;
        uint64_t size;
    };
    std::vector<Managed> managed;
    for (const auto& sym : m.symbols)
        if (sym.managed && sym.host) managed.push_back({sym.address, sym.host, sym.size});
    if (!prof.enabled && !debug) {
        // Queued on its stream; returns right away.
        auto buf = std::make_shared<std::vector<uint8_t>>(std::move(params));
        runtime().sched.enqueue(stream, [d, function, cfg, buf, name = k.name, managed]() -> cudaError_t {
            for (const Managed& v : managed) d->memcpy_htod(v.address, v.host, v.size);
            std::vector<MappedRange> copies;
            {
                std::lock_guard lock(mapped_mutex);
                for (const auto& r : mapped_ranges)
                    if (r.shadow != r.host) copies.push_back(r);
            }
            for (const auto& r : copies) std::memcpy(r.shadow, r.host, r.size);
            cpusim::LaunchResult result;
            Error e = d->launch(function, cfg, buf->data(), buf->size(), &result);
            if (kernel_times().enabled && e == Error::Success) kernel_times().add(name, result.kernel_ns);
            for (const auto& r : copies) std::memcpy(r.host, r.shadow, r.size);
            for (const Managed& v : managed) d->memcpy_dtoh(v.host, v.address, v.size);
            if (e != Error::Success && result.report[0])
                std::fprintf(stderr, "cpusim: kernel %s failed:\n  %s\n", name.c_str(), result.report);
            return static_cast<cudaError_t>(e);
        });
        return cudaSuccess;
    }
    // Profiling and debug runs report on the result: synchronous, after
    // everything issued before.
    runtime().sched.wait_all();
    if (!prof.enabled) {
        cpusim::LaunchResult result;
        Error e = d->launch(function, cfg, params.data(), params.size(), &result);
        if (e != Error::Success && result.report[0])
            std::fprintf(stderr, "cpusim: kernel %s failed:\n  %s\n", k.name.c_str(), result.report);
        return record(e);
    }

    ProfileEntry* entry;
    {
        std::lock_guard plock(prof.mutex);
        entry = prof.find(k.name, cfg);
        cfg.profile = entry == nullptr;
    }
    cpusim::LaunchResult result;
    Error e = d->launch(function, cfg, params.data(), params.size(), &result);
    if (e != Error::Success) return record(e);
    std::lock_guard plock(prof.mutex);
    if (cfg.profile) {
        ProfileEntry ne;
        ne.kernel = k.name;
        std::memcpy(ne.grid, cfg.grid, sizeof(ne.grid));
        std::memcpy(ne.block, cfg.block, sizeof(ne.block));
        std::memcpy(ne.counters, result.counters, sizeof(ne.counters));
        ne.profiled_ns = result.kernel_ns;
        prof.entries.push_back(ne);
    } else {
        entry->timed_runs += 1;
        entry->timed_ns += result.kernel_ns;
    }
    return cudaSuccess;
}

// ---- kernel attributes and occupancy

namespace {
constexpr int kResidentPerSM = 2;  // workers (logical CPUs) per multiprocessor
// The registered kernel behind a host stub, or null.
KernelDesc* kernel_of(const void* func) {
    Registry& r = registry();
    auto it = r.functions.find(func);
    return it == r.functions.end() ? nullptr : const_cast<KernelDesc*>(it->second.kernel);
}
}  // namespace

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes* attr, const void* func) {
    if (!attr) return record(Error::InvalidValue);
    REQUIRE_DEVICE(d);
    Registry& r = registry();
    std::lock_guard lock(r.mutex);
    KernelDesc* k = kernel_of(func);
    if (!k) return record(Error::InvalidDeviceFunction);
    const auto& p = d->properties();
    *attr = {};
    attr->sharedSizeBytes = k->shared_bytes;
    attr->maxThreadsPerBlock = p.maxThreadsPerBlock;
    attr->numRegs = 32;  // no registers to count: a plausible figure for occupancy arithmetic
    attr->ptxVersion = attr->binaryVersion = p.major * 10 + p.minor;
    attr->maxDynamicSharedSizeBytes =
        k->max_dynamic_shared >= 0 ? k->max_dynamic_shared : int(p.sharedMemPerBlock - k->shared_bytes);
    attr->preferredShmemCarveout = -1;
    return cudaSuccess;
}

cudaError_t cudaFuncSetAttribute(const void* func, enum cudaFuncAttribute attr, int value) {
    REQUIRE_DEVICE(d);
    Registry& r = registry();
    std::lock_guard lock(r.mutex);
    KernelDesc* k = kernel_of(func);
    if (!k) return record(Error::InvalidDeviceFunction);
    if (attr == cudaFuncAttributeMaxDynamicSharedMemorySize) {
        if (value < 0 || uint64_t(value) + k->shared_bytes > d->properties().sharedMemPerBlockOptin)
            return record(Error::InvalidValue);
        k->max_dynamic_shared = value;
    } else if (attr != cudaFuncAttributePreferredSharedMemoryCarveout) {
        return record(Error::InvalidValue);
    }
    return cudaSuccess;
}

cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCache) {
    Registry& r = registry();
    std::lock_guard lock(r.mutex);
    return kernel_of(func) ? cudaSuccess : record(Error::InvalidDeviceFunction);
}

// As a GPU counts: blocks that fit at once in one multiprocessor by threads
// (in whole warps), by shared memory and by the per-multiprocessor limit.
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* blocks, const void* func, int block_size,
                                                                   size_t dynamic_shared, unsigned int) {
    if (!blocks || block_size <= 0) return record(Error::InvalidValue);
    REQUIRE_DEVICE(d);
    uint64_t shared;
    {
        Registry& r = registry();
        std::lock_guard lock(r.mutex);
        KernelDesc* k = kernel_of(func);
        if (!k) return record(Error::InvalidDeviceFunction);
        shared = k->shared_bytes + dynamic_shared;
    }
    const auto& p = d->properties();
    if (block_size > p.maxThreadsPerBlock) {
        *blocks = 0;
        return cudaSuccess;
    }
    const int warps = (block_size + 31) / 32;
    // Blocks run whole, one per worker (logical CPU): two per multiprocessor
    // at a time, whatever the size. A program that sizes a persistent or
    // cooperative grid from this gets every block resident.
    int n = std::min({p.maxBlocksPerMultiProcessor, p.maxThreadsPerMultiProcessor / (warps * 32), kResidentPerSM});
    if (shared) n = std::min<int>(n, int(p.sharedMemPerMultiprocessor / shared));
    *blocks = std::max(n, 0);
    return cudaSuccess;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* blocks, const void* func, int block_size,
                                                          size_t dynamic_shared) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(blocks, func, block_size, dynamic_shared, 0);
}

// Every block of the grid must be resident at once (this_grid().sync()
// waits for all of them): at most the occupancy times the multiprocessors.
cudaError_t cudaLaunchCooperativeKernel(const void* func, dim3 grid, dim3 block, void** args, size_t shared,
                                        cudaStream_t stream) {
    REQUIRE_DEVICE(d);
    int per_sm = 0;
    if (cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, func, int(block.x * block.y * block.z),
                                                                      shared);
        e != cudaSuccess)
        return e;
    if (uint64_t(grid.x) * grid.y * grid.z > uint64_t(per_sm) * d->properties().multiProcessorCount)
        return record(Error::CooperativeLaunchTooLarge);
    return cudaLaunchKernel(func, grid, block, args, shared, stream);
}


}  // extern "C"
