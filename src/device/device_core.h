#pragma once

#include <cstdlib>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common/config.h"
#include "common/error.h"
#include "common/topology.h"
#include "device/allocator.h"
#include "device/amx_gemm.h"
#include "device/executor.h"
#include "device/properties.h"
#include "device/vram.h"

namespace cpusim {

// Where the driver tries to place the VRAM. Clients reserve the same range in
// their own address space, so a device pointer can never equal a valid host
// pointer (what UVA guarantees, and what cudaMemcpyDefault relies on).
// Below every range the kernel randomizes: a PIE executable lands anywhere
// from 0x555555554000 up to 16 TiB above it (vm.mmap_rnd_bits = 32), so at
// 0x600000000000 one program in fifty found its own code there, could not
// reserve the range, and saw every device pointer refused.
constexpr uint64_t kPreferredVramBase = 0x200000000000ull;

// Monotonic counters. Busy times feed the utilization numbers in cpu-smi.
struct DeviceCounters {
    std::atomic<uint64_t> kernel_busy_ns{0};
    std::atomic<uint64_t> kernel_since_ns{0};  // CLOCK_MONOTONIC start of the busy period, 0 when idle
    std::atomic<int> kernels_running{0};       // launches in progress (several with streams)
    std::atomic<uint64_t> copy_busy_ns{0};
    std::atomic<uint64_t> kernels_launched{0}, kernels_profiled{0};
    std::atomic<uint64_t> bytes_htod{0}, bytes_dtoh{0}, bytes_dtod{0}, bytes_memset{0};
    std::atomic<uint64_t> alloc_calls{0}, free_calls{0}, failed_allocs{0};
};

// Adds the lifetime of the scope to a busy counter.
class BusyScope {
public:
    explicit BusyScope(std::atomic<uint64_t>& counter)
        : counter_(counter), start_(std::chrono::steady_clock::now()) {}
    ~BusyScope() {
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start_);
        counter_.fetch_add(static_cast<uint64_t>(ns.count()), std::memory_order_relaxed);
    }

private:
    std::atomic<uint64_t>& counter_;
    std::chrono::steady_clock::time_point start_;
};

// A client of the device (one per process, like a CUDA context).
struct ContextInfo {
    uint32_t id = 0;
    int pid = 0;
    std::string process_name;
    std::chrono::system_clock::time_point started;
    uint64_t used_bytes = 0;
    uint64_t allocations = 0;
};

// The state of one simulated GPU: reserved cores, VRAM, allocator and contexts.
// Owned by the driver (daemon backend) or by the application (in-process backend).
// Thread safe.
class DeviceCore {
public:
    static std::unique_ptr<DeviceCore> create(const Config& cfg, uint64_t preferred_base, std::string* error);
    ~DeviceCore();

    const Config& config() const { return cfg_; }
    const Topology& topology() const { return topo_; }
    const DeviceProperties& properties() const { return props_; }
    const std::vector<PhysicalCore>& reserved_cores() const { return cores_; }
    std::vector<int> reserved_cpus() const;
    VramPool& vram() { return *vram_; }
    // VRAM bytes backed by host RAM (only the allocated ranges are looked at).
    uint64_t vram_resident() const;
    DeviceCounters& counters() { return counters_; }
    std::chrono::system_clock::time_point started() const { return started_; }

    uint32_t open_context(int pid, std::string process_name);
    void close_context(uint32_t ctx);  // frees everything the context still holds
    void set_output(uint32_t ctx, int fd);  // takes the descriptor
    // Stops the kernels still running for programs that are gone (or, with
    // dead_only false, every kernel): a stuck kernel of a dead program would
    // hold the device for everyone. Repeat while they last.
    void cancel_launches(bool dead_only);
    std::vector<ContextInfo> contexts() const;

    Error allocate(uint32_t ctx, uint64_t bytes, uint64_t* dptr);
    // In the shared pool (cudaMallocManaged): valid in the driver and in the
    // application at the same address. NotSupported without a pool.
    Error allocate_managed(uint32_t ctx, uint64_t bytes, uint64_t* dptr);
    // The shared pool: a memory file (fd for the clients to map) at `base`.
    int shared_fd() const { return shared_fd_; }
    uint8_t* shared_base() const { return shared_base_; }
    uint64_t shared_size() const { return shared_size_; }
    uint64_t shared_used() const;
    Error release(uint32_t ctx, uint64_t dptr);

    // Checks that [dptr, dptr + len) lies inside one allocation of `ctx` and
    // returns its address in this process.
    Error translate(uint32_t ctx, uint64_t dptr, uint64_t len, uint8_t** host) const;

    Error memset(uint32_t ctx, uint64_t dptr, int value, uint64_t n);

    // Loads a kernel module (a native shared object) from a file descriptor.
    // The handle is only valid for this context.
    Error load_module(uint32_t ctx, int fd, uint64_t* module);
    Error get_function(uint32_t ctx, uint64_t module, const std::string& name, uint64_t* function);
    Error bind_symbols(uint32_t ctx, uint64_t module, const std::string& table, bool init);
    Error launch(uint32_t ctx, uint64_t function, const LaunchConfig& cfg, const void* params, size_t param_bytes,
                 LaunchResult* result);
    Error memcpy_dtod(uint32_t ctx, uint64_t dst, uint64_t src, uint64_t n);

    void mem_info(uint64_t* free, uint64_t* total, uint64_t* largest_free = nullptr) const;

private:
    Error allocate_in(RangeAllocator& pool, uint8_t* base, uint32_t ctx, uint64_t bytes, uint64_t* dptr);
    bool in_shared(uint64_t p) const {
        return shared_base_ && p >= uint64_t(shared_base_) && p < uint64_t(shared_base_) + shared_size_;
    }
    void release_range(uint64_t dptr);  // back to its pool; the pages to the OS
    int shared_fd_ = -1;
    uint8_t* shared_base_ = nullptr;
    uint64_t shared_size_ = 0;
    std::unique_ptr<RangeAllocator> shared_allocator_;

    struct Context {
        ContextInfo info;
        std::map<uint64_t, uint64_t> allocations;  // device pointer -> size
        std::vector<void*> modules;                // dlopen handles
        int running = 0;                           // launches in flight
        int output_fd = 1;                         // device printf (owned unless 1)
        // Per block shape: which width runs fastest, measured on the first launches.
        struct Tuning {
            std::vector<uint32_t> candidates;  // lane widths still to try, then the winner
            std::vector<uint64_t> ns;
            size_t first_round = 0;  // candidates in the first round, once it is done
            uint32_t chosen = 0;
        };
        struct FunctionInfo {
            uint64_t profiled = 0;  // instrumented twin
            // With W the warp size (cpusim_kernel_meta.warp_size):
            uint64_t wide[5] = {};  // variants with 2W << i lanes (64..512 for warps of 32)
            uint64_t susp[2] = {};  // suspending variants with 2W and 4W lanes
            uint64_t block = 0;     // __cpusim_block_: whole block, 32-lane warps, barrier rounds inside
            uint64_t trace = 0;     // __cpusim_trace_: a wmma GEMM candidate (wmma_trace.h)
            uint64_t refill = 0;    // __cpusim_entryr_: warp entry whose lanes refill (CPUSIM_META_REFILL)
            std::string name;       // the kernel's (mangled), for messages
            cpusim_kernel_meta meta{};
            std::map<uint64_t, Tuning> tuning;  // key: block shape
            // Launches of a wmma GEMM candidate, by grid, block and
            // parameters: the GEMM they compute (type 0: not one).
            std::map<std::string, GemmArgs> gemm_plans;
        };
        std::map<uint64_t, FunctionInfo> functions;  // entry point handed out
    };

    DeviceCore() = default;
    // Blocks per candidate when the width is measured inside one launch: 32
    // per worker (CPUSIM_TRIAL_BLOCKS), a few ms per slice. Shorter slices
    // measured clock and cache noise more than the width (matmul_tiled: 8 per
    // worker picked a variant 40% slower over the whole launch).
    uint64_t trial_blocks() const {
        static const uint64_t per_worker = [] {
            const char* v = std::getenv("CPUSIM_TRIAL_BLOCKS");
            return v ? std::max(1, std::atoi(v)) : 32;
        }();
        return per_worker * executor_->workers();
    }
    static constexpr int kTrialRounds = 2;
    // A width to try, with its entry point.
    struct Trial {
        uint32_t lanes;
        uint64_t entry;
    };
    uint32_t pick_width(Context::FunctionInfo& fn, const LaunchConfig& cfg, uint64_t* entry, uint64_t* tuning_key,
                        std::vector<Trial>* trials);
    void choose_width(uint32_t ctx, uint64_t function, uint64_t key, const std::vector<Trial>& trials,
                      const std::vector<uint64_t>& ns);
    void record_timing(uint32_t ctx, uint64_t function, uint64_t key, uint32_t lanes, uint64_t ns);

    Config cfg_;
    Topology topo_;
    std::vector<PhysicalCore> cores_;
    DeviceProperties props_{};
    std::unique_ptr<VramPool> vram_;
    std::unique_ptr<RangeAllocator> allocator_;
    DeviceCounters counters_;
    // Created on the first launch: worker threads must not exist before the
    // driver detaches (fork keeps only the calling thread).
    std::unique_ptr<Executor> executor_;
    std::once_flag executor_once_;
    std::chrono::system_clock::time_point started_;

    mutable std::mutex mutex_;
    std::map<uint32_t, Context> contexts_;
    uint32_t next_context_ = 1;
};

}  // namespace cpusim
