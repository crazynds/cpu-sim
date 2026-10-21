#include "device/device_core.h"

#include "device/wmma_trace.h"

#include <sys/mman.h>
#include "device/inprocess_status.h"

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <sstream>
#include <thread>
#include <fstream>

#include "common/units.h"
#include "device/debug_exec.h"

namespace cpusim {

namespace {

uint64_t host_memory_total() {
    std::ifstream f("/proc/meminfo");
    std::string key;
    uint64_t kb = 0;
    while (f >> key >> kb) {
        if (key == "MemTotal:") return kb * KiB;
        f.ignore(256, '\n');
    }
    return 0;
}

// Picks the physical cores to reserve. Explicit `cpus` wins; otherwise the last
// `cores` physical cores, leaving the first ones (where most IRQs and the
// application's host threads land) to the OS.
bool pick_cores(const Config& cfg, const Topology& topo, std::vector<PhysicalCore>* out, std::string* error) {
    out->clear();
    if (!cfg.cpus.empty()) {
        for (const auto& core : topo.cores) {
            bool wanted = std::any_of(core.cpus.begin(), core.cpus.end(), [&](int c) {
                return std::find(cfg.cpus.begin(), cfg.cpus.end(), c) != cfg.cpus.end();
            });
            if (wanted) out->push_back(core);
        }
        if (out->empty()) {
            *error = "none of the configured cpus (" + format_cpu_list(cfg.cpus) + ") is online";
            return false;
        }
        return true;
    }

    int available = static_cast<int>(topo.cores.size());
    int wanted = cfg.cores > 0 ? cfg.cores : std::max(1, available - 2);
    if (wanted > available) {
        *error = "configured " + std::to_string(wanted) + " cores, but the machine has " + std::to_string(available) +
                 " physical cores";
        return false;
    }
    out->assign(topo.cores.end() - wanted, topo.cores.end());
    return true;
}

}  // namespace

std::unique_ptr<DeviceCore> DeviceCore::create(const Config& cfg, uint64_t preferred_base, std::string* error) {
    std::string err;
    if (!cpu_has_avx512()) {
        if (error) *error = "this CPU does not support AVX-512 (F, BW, DQ, VL), which the generated code requires";
        return nullptr;
    }
    std::unique_ptr<DeviceCore> dev(new DeviceCore);
    dev->cfg_ = cfg;
    dev->topo_ = read_topology();
    if (!pick_cores(cfg, dev->topo_, &dev->cores_, &err)) {
        if (error) *error = err;
        return nullptr;
    }

    uint64_t vram = cfg.vram_private;
    if (vram == 0) {
        // A quarter of the host RAM, at most 8 GiB, rounded down to whole GiB.
        vram = std::clamp<uint64_t>(host_memory_total() / 4 / GiB * GiB, GiB, 8 * GiB);
    }
    dev->vram_ = VramPool::create(vram, cfg.hugepages, cfg.prefault, preferred_base, &err);
    if (!dev->vram_) {
        if (error) *error = err;
        return nullptr;
    }
    dev->allocator_ = std::make_unique<RangeAllocator>(dev->vram_->size());

    if (cfg.vram_shared > 0) {
        // The shared pool: a memory file the clients map at the same address.
        // Right after the VRAM, 1 GiB apart, when the VRAM has a fixed place
        // (the driver), so the range is free in the clients too.
        const uint64_t size = (cfg.vram_shared + (2 << 20) - 1) / (2 << 20) * (2 << 20);
        int fd = memfd_create("cpusim-shared", MFD_CLOEXEC);
        if (fd < 0 || ftruncate(fd, off_t(size)) != 0) {
            if (error) *error = std::string("shared pool: ") + std::strerror(errno);
            if (fd >= 0) close(fd);
            return nullptr;
        }
        void* p = MAP_FAILED;
        if (preferred_base) {
            uint64_t want = (reinterpret_cast<uint64_t>(dev->vram_->base()) + dev->vram_->size() + 2 * GiB - 1) / GiB * GiB;
            p = mmap(reinterpret_cast<void*>(want), size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED_NOREPLACE, fd, 0);
        }
        if (p == MAP_FAILED) p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) {
            if (error) *error = std::string("shared pool: mmap: ") + std::strerror(errno);
            close(fd);
            return nullptr;
        }
        dev->shared_fd_ = fd;
        dev->shared_base_ = static_cast<uint8_t*>(p);
        dev->shared_size_ = size;
        dev->shared_allocator_ = std::make_unique<RangeAllocator>(size);
    }

    int max_khz = 0;
    for (int cpu : dev->reserved_cpus()) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/cpuinfo_max_freq");
        int khz = 0;
        if (f >> khz) max_khz = std::max(max_khz, khz);
    }
    dev->props_ = make_properties(cfg, dev->topo_, static_cast<int>(dev->cores_.size()), dev->vram_->size(), max_khz);
    dev->started_ = std::chrono::system_clock::now();
    return dev;
}

std::vector<int> DeviceCore::reserved_cpus() const {
    std::vector<int> cpus;
    for (const auto& core : cores_) cpus.insert(cpus.end(), core.cpus.begin(), core.cpus.end());
    std::sort(cpus.begin(), cpus.end());
    return cpus;
}

uint32_t DeviceCore::open_context(int pid, std::string process_name) {
    std::lock_guard lock(mutex_);
    uint32_t id = next_context_++;
    Context& ctx = contexts_[id];
    ctx.info.id = id;
    ctx.info.pid = pid;
    ctx.info.process_name = std::move(process_name);
    ctx.info.started = std::chrono::system_clock::now();
    return id;
}

DeviceCore::~DeviceCore() {
    if (shared_base_) munmap(shared_base_, shared_size_);
    if (shared_fd_ >= 0) close(shared_fd_);
}

void DeviceCore::release_range(uint64_t dptr) {
    uint64_t begin = 0, end = 0;
    if (in_shared(dptr)) {
        shared_allocator_->release(dptr - reinterpret_cast<uint64_t>(shared_base_), &begin, &end);
        if (end > begin) madvise(shared_base_ + begin, end - begin, MADV_REMOVE);  // the pages back to the OS
        return;
    }
    allocator_->release(dptr - reinterpret_cast<uint64_t>(vram_->base()), &begin, &end);
    vram_->discard(begin, end);
}

uint64_t DeviceCore::shared_used() const {
    std::lock_guard lock(mutex_);
    return shared_allocator_ ? shared_allocator_->used() : 0;
}

void DeviceCore::set_output(uint32_t id, int fd) {
    std::lock_guard lock(mutex_);
    auto it = contexts_.find(id);
    if (it == contexts_.end()) {
        close(fd);
        return;
    }
    if (it->second.output_fd != 1) close(it->second.output_fd);
    it->second.output_fd = fd;
}

void DeviceCore::close_context(uint32_t id) {
    // Its kernels still running (the program died with them in flight) stop
    // first: they use its memory and its modules' code.
    while (true) {
        {
            std::lock_guard lock(mutex_);
            auto it = contexts_.find(id);
            if (it == contexts_.end()) return;
            if (it->second.running == 0) break;
        }
        if (executor_) executor_->cancel(id);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::lock_guard lock(mutex_);
    auto it = contexts_.find(id);
    if (it == contexts_.end()) return;
    for (auto [ptr, size] : it->second.allocations) release_range(ptr);
    for (void* handle : it->second.modules) dlclose(handle);
    if (it->second.output_fd != 1) close(it->second.output_fd);
    contexts_.erase(it);
}

std::vector<ContextInfo> DeviceCore::contexts() const {
    std::lock_guard lock(mutex_);
    std::vector<ContextInfo> out;
    for (const auto& [id, ctx] : contexts_) out.push_back(ctx.info);
    return out;
}

Error DeviceCore::allocate(uint32_t id, uint64_t bytes, uint64_t* dptr) {
    return allocate_in(*allocator_, vram_->base(), id, bytes, dptr);
}

Error DeviceCore::allocate_managed(uint32_t id, uint64_t bytes, uint64_t* dptr) {
    if (!shared_allocator_) return Error::NotSupported;
    return allocate_in(*shared_allocator_, shared_base_, id, bytes, dptr);
}

Error DeviceCore::allocate_in(RangeAllocator& pool, uint8_t* pool_base, uint32_t id, uint64_t bytes, uint64_t* dptr) {
    counters_.alloc_calls.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(mutex_);
    auto it = contexts_.find(id);
    if (it == contexts_.end()) return Error::InitializationError;
    if (bytes == 0) {
        *dptr = 0;
        return Error::Success;
    }
    auto off = pool.allocate(bytes);
    if (!off) {
        counters_.failed_allocs.fetch_add(1, std::memory_order_relaxed);
        return Error::MemoryAllocation;
    }
    auto [start, size] = *pool.find(*off, 0);
    *dptr = reinterpret_cast<uint64_t>(pool_base) + start;
    if (&pool == allocator_.get()) vram_->populate(start, start + size);
    it->second.allocations.emplace(*dptr, size);
    it->second.info.used_bytes += size;
    it->second.info.allocations += 1;
    return Error::Success;
}

Error DeviceCore::release(uint32_t id, uint64_t dptr) {
    counters_.free_calls.fetch_add(1, std::memory_order_relaxed);
    if (dptr == 0) return Error::Success;  // cudaFree(nullptr) is a no-op
    std::lock_guard lock(mutex_);
    auto it = contexts_.find(id);
    if (it == contexts_.end()) return Error::InitializationError;
    auto a = it->second.allocations.find(dptr);
    if (a == it->second.allocations.end()) return Error::InvalidValue;

    release_range(dptr);
    it->second.info.used_bytes -= a->second;
    it->second.info.allocations -= 1;
    it->second.allocations.erase(a);
    return Error::Success;
}

Error DeviceCore::translate(uint32_t id, uint64_t dptr, uint64_t len, uint8_t** host) const {
    std::lock_guard lock(mutex_);
    auto it = contexts_.find(id);
    if (it == contexts_.end()) return Error::InitializationError;
    const auto& allocs = it->second.allocations;
    auto a = allocs.upper_bound(dptr);
    if (a == allocs.begin()) return Error::InvalidValue;
    --a;
    uint64_t offset = dptr - a->first;
    if (offset >= a->second || len > a->second - offset) return Error::InvalidValue;
    *host = reinterpret_cast<uint8_t*>(dptr);
    return Error::Success;
}

Error DeviceCore::memset(uint32_t id, uint64_t dptr, int value, uint64_t n) {
    if (n == 0) return Error::Success;
    uint8_t* p = nullptr;
    if (Error e = translate(id, dptr, n, &p); e != Error::Success) return e;
    BusyScope busy(counters_.copy_busy_ns);
    std::memset(p, value, n);
    counters_.bytes_memset.fetch_add(n, std::memory_order_relaxed);
    return Error::Success;
}

Error DeviceCore::memcpy_dtod(uint32_t id, uint64_t dst, uint64_t src, uint64_t n) {
    if (n == 0) return Error::Success;
    uint8_t *d = nullptr, *s = nullptr;
    if (Error e = translate(id, dst, n, &d); e != Error::Success) return e;
    if (Error e = translate(id, src, n, &s); e != Error::Success) return e;
    BusyScope busy(counters_.copy_busy_ns);
    std::memmove(d, s, n);
    counters_.bytes_dtod.fetch_add(n, std::memory_order_relaxed);
    return Error::Success;
}

namespace {


// Kernel modules are loaded from a real file named after their content, so
// profilers (perf report/annotate) can find the code and its symbols after
// the run. Same build, same file: nothing accumulates across runs.
std::string module_file(int fd, uint64_t* hash) {
    std::string data;
    char buf[1 << 16];
    ssize_t n;
    off_t off = 0;
    while ((n = pread(fd, buf, sizeof(buf), off)) > 0) {
        data.append(buf, size_t(n));
        off += n;
    }
    uint64_t h = 1469598103934665603ull;  // FNV-1a
    for (unsigned char c : data) h = (h ^ c) * 1099511628211ull;
    *hash = h;

    const char* rt = std::getenv("XDG_RUNTIME_DIR");
    std::string dir = std::string(rt && *rt ? rt : "/tmp") + "/cpusim";
    mkdir(dir.c_str(), 0700);
    dir += "/modules";
    mkdir(dir.c_str(), 0700);
    char name[64];
    std::snprintf(name, sizeof(name), "/%016llx.so", (unsigned long long)h);
    std::string path = dir + name;
    if (access(path.c_str(), R_OK) != 0) {
        std::string tmp = path + "." + std::to_string(getpid());
        if (FILE* f = std::fopen(tmp.c_str(), "wb")) {
            bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size();
            ok = (std::fclose(f) == 0) && ok;
            if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
                std::remove(tmp.c_str());
                return "/proc/self/fd/" + std::to_string(fd);
            }
        } else {
            return "/proc/self/fd/" + std::to_string(fd);
        }
    }
    return path;
}

}  // namespace

Error DeviceCore::load_module(uint32_t id, int fd, uint64_t* module) {
    uint64_t hash = 0;
    std::string path = module_file(fd, &hash);
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::fprintf(stderr, "cpusim: cannot load kernel module: %s\n", dlerror());
        return Error::InvalidKernelImage;
    }
    auto* abi = static_cast<const uint32_t*>(dlsym(handle, "__cpusim_abi_version"));
    if (!abi || *abi != CPUSIM_KERNEL_ABI) {
        std::fprintf(stderr, "cpusim: kernel module ABI %u, runtime expects %u\n", abi ? *abi : 0u, CPUSIM_KERNEL_ABI);
        dlclose(handle);
        return Error::InvalidKernelImage;
    }
    std::lock_guard lock(mutex_);
    auto it = contexts_.find(id);
    if (it == contexts_.end()) {
        dlclose(handle);
        return Error::InitializationError;
    }
    it->second.modules.push_back(handle);
    it->second.module_code[handle] = hash;
    *module = reinterpret_cast<uint64_t>(handle);
    return Error::Success;
}

// What a faulted kernel's launch returns: an invalid address, a trap (device
// assert, __trap), anything else.
static Error fault_error(int signal) {
    if (signal == SIGSEGV || signal == SIGBUS) return Error::IllegalAddress;
    if (signal == SIGILL) return Error::Assert;
    return Error::LaunchFailure;
}

void DeviceCore::cancel_launches(bool dead_only) {
    if (!executor_) return;
    std::vector<uint32_t> ids;
    {
        std::lock_guard lock(mutex_);
        for (const auto& [id, c] : contexts_)
            if (!dead_only || (kill(c.info.pid, 0) != 0 && errno == ESRCH)) ids.push_back(id);
    }
    if (!dead_only) ids = {0};
    for (uint32_t id : ids) executor_->cancel(id);
}

Error DeviceCore::bind_symbols(uint32_t id, uint64_t module, const std::string& table, bool init) {
    std::lock_guard lock(mutex_);
    auto it = contexts_.find(id);
    if (it == contexts_.end()) return Error::InitializationError;
    auto& mods = it->second.modules;
    void* handle = reinterpret_cast<void*>(module);
    if (std::find(mods.begin(), mods.end(), handle) == mods.end()) return Error::InvalidResourceHandle;
    const uint8_t* lo = vram().base();
    const uint8_t* hi = lo + vram().size();
    std::istringstream in(table);
    uint64_t address, size;
    std::string name;
    while (in >> address >> size >> name) {
        auto* dst = reinterpret_cast<uint8_t*>(address);
        if (dst < lo || dst + size > hi) return Error::InvalidValue;
        // A module without the variable does not use it (dropped as dead).
        if (auto* slot = static_cast<void**>(dlsym(handle, ("__cpusim_sym_" + name).c_str()))) *slot = dst;
        if (init)
            if (const void* src = dlsym(handle, ("__cpusim_init_" + name).c_str())) std::memcpy(dst, src, size);
    }
    return Error::Success;
}

Error DeviceCore::get_function(uint32_t id, uint64_t module, const std::string& name, uint64_t* function) {
    std::lock_guard lock(mutex_);
    auto it = contexts_.find(id);
    if (it == contexts_.end()) return Error::InitializationError;
    auto& mods = it->second.modules;
    void* handle = reinterpret_cast<void*>(module);
    if (std::find(mods.begin(), mods.end(), handle) == mods.end()) return Error::InvalidResourceHandle;
    void* sym = dlsym(handle, ("__cpusim_entry_" + name).c_str());
    void* prof = dlsym(handle, ("__cpusim_prof_" + name).c_str());
    auto* meta = static_cast<const cpusim_kernel_meta*>(dlsym(handle, ("__cpusim_meta_" + name).c_str()));
    if (!sym || !meta) return Error::InvalidDeviceFunction;
    *function = reinterpret_cast<uint64_t>(sym);
    Context::FunctionInfo info;
    info.name = name;
    // The code it runs: its module's content and its name.
    info.code = it->second.module_code[handle];
    for (unsigned char c : name) info.code = (info.code ^ c) * 1099511628211ull;
    info.profiled = reinterpret_cast<uint64_t>(prof);
    info.meta = *meta;
    const uint32_t two_warps = 2 * meta->warp_size;  // the narrowest wide variant
    for (int i = 0; i < 5; ++i)
        if (meta->flags & (1u << (CPUSIM_META_WIDE_SHIFT + i)))
            info.wide[i] = reinterpret_cast<uint64_t>(
                dlsym(handle, ("__cpusim_entry" + std::to_string(two_warps << i) + "_" + name).c_str()));
    for (int i = 0; i < 2; ++i)
        if (meta->flags & (1u << (CPUSIM_META_SUSP_SHIFT + i)))
            info.susp[i] = reinterpret_cast<uint64_t>(
                dlsym(handle, ("__cpusim_entry" + std::to_string(two_warps << i) + "s_" + name).c_str()));
    if (meta->flags & CPUSIM_META_BLOCK_FN)
        info.block = reinterpret_cast<uint64_t>(dlsym(handle, ("__cpusim_block_" + name).c_str()));
    info.trace = reinterpret_cast<uint64_t>(dlsym(handle, ("__cpusim_trace_" + name).c_str()));
    if (meta->flags & CPUSIM_META_REFILL)
        info.refill = reinterpret_cast<uint64_t>(dlsym(handle, ("__cpusim_entryr_" + name).c_str()));
    it->second.functions[*function] = info;
    return Error::Success;
}

Error DeviceCore::launch(uint32_t id, uint64_t function, const LaunchConfig& cfg, const void* params,
                         size_t param_bytes, LaunchResult* result) {
    uint64_t entry = function;
    uint64_t block_fn = 0;
    uint32_t lanes = 0;  // the warp entry's (meta.warp_lanes) unless a wider variant is picked
    uint64_t tuning_key = 0;
    std::vector<Trial> trials;  // width tuning inside this launch
    int output_fd = 1;
    uint64_t huge_lo = 0, huge_hi = UINT64_MAX;
    uint64_t trace_fn = 0;
    cpusim_kernel_meta meta{};
    // The executor first: the widths offered depend on its pairs of SMT siblings.
    std::call_once(executor_once_, [this] { executor_ = std::make_unique<Executor>(reserved_cpus()); });
    {
        std::lock_guard lock(mutex_);
        auto it = contexts_.find(id);
        if (it == contexts_.end()) return Error::InitializationError;
        auto fn = it->second.functions.find(function);
        if (fn == it->second.functions.end()) return Error::InvalidDeviceFunction;
        meta = fn->second.meta;
        lanes = meta.warp_lanes;
        trace_fn = fn->second.trace;
        ++it->second.running;
        output_fd = it->second.output_fd;
        // The span of the allocations of 2 GiB or more, in VRAM and in the
        // shared pool (their hull when both have some). CPUSIM_SMALL_OFFSETS=0:
        // everything, so 64-bit gather indices into global memory (comparisons).
        static const bool allow_small = [] {
            const char* v = std::getenv("CPUSIM_SMALL_OFFSETS");
            return !v || std::atoi(v) != 0;
        }();
        if (allow_small) {
            huge_lo = UINT64_MAX;
            huge_hi = 0;
            auto add = [&](const RangeAllocator* a, const uint8_t* base) {
                if (!a) return;
                if (auto span = a->huge_span()) {
                    huge_lo = std::min(huge_lo, reinterpret_cast<uint64_t>(base) + span->first);
                    huge_hi = std::max(huge_hi, reinterpret_cast<uint64_t>(base) + span->second);
                }
            };
            add(allocator_.get(), vram().base());
            add(shared_allocator_.get(), shared_base_);
        }
        if (cfg.profile) {
            if (!fn->second.profiled) return Error::NotSupported;
            entry = fn->second.profiled;
        } else if (!cfg.debug) {
            lanes = pick_width(fn->second, cfg, &entry, &tuning_key, &trials);
            // Opt-in: measured slower than per-warp calls, and it does not use
            // barrier_pass (the last warp to arrive still suspends).
            static const bool use_block = std::getenv("CPUSIM_BLOCK_FN") != nullptr;
            if (lanes == meta.warp_lanes && use_block) block_fn = fn->second.block;
        }
    }
    struct Running {  // the launch counts as running until it returns
        DeviceCore& core;
        uint32_t id;
        ~Running() {
            std::lock_guard lock(core.mutex_);
            if (auto it = core.contexts_.find(id); it != core.contexts_.end()) --it->second.running;
        }
    } running{*this, id};
    const auto& p = props_;
    uint64_t threads = uint64_t(cfg.block[0]) * cfg.block[1] * cfg.block[2];
    for (int i = 0; i < 3; ++i)
        if (cfg.grid[i] == 0 || cfg.block[i] == 0 || cfg.block[i] > uint32_t(p.maxThreadsDim[i]) ||
            cfg.grid[i] > uint32_t(p.maxGridSize[i]))
            return Error::InvalidConfiguration;
    if (threads > uint64_t(p.maxThreadsPerBlock)) return Error::InvalidConfiguration;
    // Dynamic shared memory goes right after the kernel's static __shared__ variables.
    LaunchConfig run_cfg = cfg;
    run_cfg.output_fd = output_fd;
    run_cfg.huge_lo = huge_lo;
    run_cfg.huge_hi = huge_hi;
    run_cfg.shared_bytes = meta.shared_bytes + cfg.shared_bytes;
    if (run_cfg.shared_bytes > Executor::kMaxSharedBytes) return Error::InvalidValue;

    // The parameter buffer is shared by every warp; keep a private, aligned copy.
    std::vector<uint8_t> buffer(std::max<size_t>(param_bytes, 1) + 64);
    auto* aligned = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(buffer.data()) + 63) & ~uintptr_t(63));
    std::memcpy(aligned, params, param_bytes);

    // A matrix multiplication written with wmma: on AMX tiles instead, once
    // the trace of the launch shows that it is one (wmma_trace.h).
    if (trace_fn && !cfg.profile && !cfg.debug &&
        (amx_gemm_available(GemmType::BF16_F32) || amx_gemm_available(GemmType::S8_S32))) {
        std::string key(reinterpret_cast<const char*>(cfg.grid), sizeof(cfg.grid));
        key.append(reinterpret_cast<const char*>(cfg.block), sizeof(cfg.block));
        key.append(reinterpret_cast<const char*>(params), param_bytes);
        GemmArgs plan;
        bool known = false;
        {
            std::lock_guard lock(mutex_);
            auto it = contexts_.find(id);
            if (it != contexts_.end())
                if (auto fn = it->second.functions.find(function); fn != it->second.functions.end())
                    if (auto p = fn->second.gemm_plans.find(key); p != fn->second.gemm_plans.end()) plan = p->second, known = true;
        }
        if (!known) {
            const char* why = "";
            const auto t0 = std::chrono::steady_clock::now();
            if (!wmma_gemm_plan(*executor_, reinterpret_cast<cpusim_trace_fn>(trace_fn), run_cfg, aligned, &plan, &why) ||
                !amx_gemm_available(plan.type))
                plan = GemmArgs{}, plan.M = 0;
            if (std::getenv("CPUSIM_VERBOSE"))
                std::fprintf(stderr, "cpusim: wmma GEMM %s: %s (traced in %.1f ms)\n", plan.M ? "recognized" : "not recognized",
                             plan.M ? (plan.type == GemmType::S8_S32 ? "int8" : "bf16") : why,
                             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            std::lock_guard lock(mutex_);
            auto it = contexts_.find(id);
            if (it != contexts_.end())
                if (auto fn = it->second.functions.find(function); fn != it->second.functions.end())
                    fn->second.gemm_plans[key] = plan;
        }
        if (plan.M) {
            const auto start = std::chrono::steady_clock::now();
            KernelFault fault = amx_gemm(*executor_, plan);
            if (fault.signal) return fault_error(fault.signal);
            LaunchResult local;
            if (!result) result = &local;
            result->kernel_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now() - start).count());
            counters_.kernels_launched.fetch_add(1, std::memory_order_relaxed);
            return Error::Success;
        }
    }

    LaunchResult local;
    if (!result) result = &local;
    uint64_t launch_busy = 0;  // workers' time on this launch's blocks
    uint64_t launch_span = 0;  // the most CPU time of one of them (Executor::run's span_ns)
    uint64_t launch_cpu = 0;   // their CPU time
    {
        // Busy time = time with at least one kernel running (launches from
        // several streams overlap). `since` lets the monitor count a long
        // kernel while it runs, not only when it ends.
        struct BusyPeriod {
            DeviceCounters& c;
            explicit BusyPeriod(DeviceCounters& counters) : c(counters) {
                if (c.kernels_running.fetch_add(1) == 0) c.kernel_since_ns.store(monotonic_ns());
            }
            ~BusyPeriod() {
                if (c.kernels_running.fetch_sub(1) != 1) return;
                uint64_t since = c.kernel_since_ns.exchange(0);
                if (since) c.kernel_busy_ns.fetch_add(monotonic_ns() - since);
            }
        } busy{counters_};
        auto start = std::chrono::steady_clock::now();
        if (cfg.debug) {
            std::string report;
            bool ok = run_debug(reinterpret_cast<cpusim_entry_fn>(entry), meta, run_cfg, aligned,
                                cfg.profile ? result->counters : nullptr, cfg.debug >= 2, &report);
            if (!ok) {
                std::fprintf(stderr, "cpusim debug: %s\n", report.c_str());
                std::snprintf(result->report, sizeof(result->report), "%s", report.c_str());
                return Error::IllegalAddress;
            }
        } else {
            KernelFault fault;
            auto run = [&](uint64_t fn_entry, uint32_t w, uint64_t first, uint64_t last, uint64_t* busy,
                           uint64_t* span = nullptr, uint64_t* cpu = nullptr) {
                if (fault.signal) return;
                executor_->run(reinterpret_cast<cpusim_entry_fn>(fn_entry), w & 0xffff,
                               w == meta.warp_lanes || (w & kSuspend), meta, run_cfg, aligned,
                               cfg.profile ? result->counters : nullptr, reinterpret_cast<cpusim_block_fn>(block_fn),
                               first, last, busy, &fault, id, (w & kRefill) != 0, span, cpu);
            };
            // A fault ends the launch (and only it): an invalid address is
            // cudaErrorIllegalAddress, anything else an unspecified failure.
            auto failed = [&]() {
                if (!fault.signal) return false;
                std::snprintf(result->report, sizeof(result->report), "%s at address 0x%lx, instruction %s",
                              strsignal(fault.signal), (unsigned long)fault.address, where_in_code(fault.ip).c_str());
                return true;
            };
            uint64_t first = 0;
            if (!trials.empty()) {
                // One slice of blocks per candidate, timed per block; then the
                // rest of the launch with the winner.
                const uint64_t slice = trial_blocks();
                std::vector<uint64_t> ns;
                for (const Trial& tr : trials) {
                    uint64_t busy = 0, cpu = 0;
                    run(tr.entry, tr.lanes, first, first + slice, &busy, nullptr, &cpu);
                    ns.push_back(cpu * 1000 / slice);
                    first += slice;
                }
                // Each candidate's slices (trials are the rounds back to back).
                const size_t n = trials.size() / trial_rounds();
                std::vector<std::vector<uint64_t>> slices(n);
                for (size_t k = 0; k < trials.size(); ++k) slices[k % n].push_back(ns[k]);
                if (failed()) return fault_error(fault.signal);
                lanes = choose_width(id, function, tuning_key, std::vector<Trial>(trials.begin(), trials.begin() + n),
                                     slices);
                for (size_t i = 0; i < n; ++i)
                    if (trials[i].lanes == lanes) entry = trials[i].entry;
            }
            run(entry, lanes, first, UINT64_MAX, &launch_busy, &launch_span, &launch_cpu);
            if (failed()) return fault_error(fault.signal);
        }
        result->kernel_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - start).count());
    }
    // CPUSIM_VERBOSE=2: how busy the workers were (sum of their time on this
    // launch over workers x wall time), to tell load imbalance from slow code.
    static const bool verbose_launches = [] {
        const char* v = std::getenv("CPUSIM_VERBOSE");
        return v && std::atoi(v) >= 2;
    }();
    if (verbose_launches && result->kernel_ns)
        std::fprintf(stderr, "cpusim: launch %u%s lanes, %.3f ms, workers busy %.0f%%\n", lanes & 0xffff,
                     width_suffix(lanes), result->kernel_ns / 1e6,
                     100.0 * double(launch_busy) / (double(result->kernel_ns) * double(executor_->workers())));
    // One candidate per launch (launches too small for in-launch trials):
    // whole launches of one shape compared with each other, by their
    // critical path, the most CPU time a worker spent on them. Like the wall time
    // it counts how well the width spreads over the workers (matrix-rotate:
    // 128 lanes does less work in all but leaves cores idle); unlike it, it
    // leaves out the round trip and the workers' wake-up, whose noise made the
    // same width measure 2-4x apart from one launch to the next. Per block,
    // for grids that vary.
    const uint64_t launch_blocks = uint64_t(cfg.grid[0]) * cfg.grid[1] * cfg.grid[2];
    if (tuning_key && trials.empty() && launch_span)
        record_timing(id, function, tuning_key, lanes, launch_span * 1000 / std::max<uint64_t>(launch_blocks, 1),
                      launch_cpu);
    counters_.kernels_launched.fetch_add(1, std::memory_order_relaxed);
    if (cfg.profile) counters_.kernels_profiled.fetch_add(1, std::memory_order_relaxed);
    return Error::Success;
}

// The width of a launch: forced (CPUSIM_LANES=<n>, <n>s or <n>r, when valid),
// the one measured fastest for this kernel and block shape, or the next one
// to measure (valid_widths, WidthTuning).
uint32_t DeviceCore::pick_width(Context::FunctionInfo& fn, const LaunchConfig& cfg, uint64_t* entry,
                                uint64_t* tuning_key, std::vector<Trial>* trials) {
    static const uint32_t forced = parse_width(std::getenv("CPUSIM_LANES"));
    const uint64_t threads = uint64_t(cfg.block[0]) * cfg.block[1] * cfg.block[2];
    const uint32_t warp = fn.meta.warp_lanes;     // the warp entry
    const uint32_t two = 2 * fn.meta.warp_size;   // the narrowest wide variant
    auto entry_for = [&](uint32_t w) -> uint64_t {
        if (w == warp) return *entry;
        if (w == (warp | kRefill)) return fn.refill;
        for (int i = 0; i < 2; ++i)
            if ((two << i | kSuspend) == w) return fn.susp[i];
        for (int i = 0; i < 5; ++i)
            if (two << i == w) return fn.wide[i];
        return 0;
    };
    KernelVariants v;
    v.warp_lanes = warp;
    v.warp_size = fn.meta.warp_size;
    v.barriers = fn.meta.flags & CPUSIM_META_HAS_BARRIERS;
    for (int i = 0; i < 2; ++i) v.susp[i] = fn.susp[i] != 0;
    for (int i = 0; i < 5; ++i) v.wide[i] = fn.wide[i] != 0;
    v.refill = fn.refill != 0;
    const std::vector<uint32_t> valid = valid_widths(v, threads);
    if (forced) {
        for (uint32_t w : valid)
            if (w == forced) {
                *entry = entry_for(w);
                return w;
            }
        return warp;
    }
    if (valid.size() == 1) return warp;

    uint64_t key = (uint64_t(cfg.block[0]) << 40) ^ (uint64_t(cfg.block[1]) << 20) ^ cfg.block[2];
    WidthTuning* tp = tuning(fn, key);
    if (!tp) return warp;
    WidthTuning& t = *tp;
    if (!t.started()) {
        // The measure is the workers' time per block (Executor::run), so it
        // holds for small slices and small launches alike.
        t.start(valid);
        // A launch with enough blocks measures every candidate on slices of
        // its own blocks and runs the rest with the winner: the choice is made
        // in the first launch. Smaller launches measure one candidate per
        // launch (every launch counts, whatever its size).
        const uint64_t blocks = uint64_t(cfg.grid[0]) * cfg.grid[1] * cfg.grid[2];
        if (blocks >= 2 * trial_rounds() * trial_blocks() * valid.size()) {
            for (int r = 0; r < trial_rounds(); ++r)
                for (uint32_t w : valid) trials->push_back({w, entry_for(w)});
            *tuning_key = key;
            *entry = trials->front().entry;
            return trials->front().lanes;
        }
    }
    uint32_t w = t.chosen() ? t.chosen() : t.next();
    if (!t.chosen()) *tuning_key = key;
    *entry = entry_for(w);
    return w;
}

WidthTuning* DeviceCore::tuning(const Context::FunctionInfo& fn, uint64_t shape) {
    const WidthKey key{fn.code, shape, uint32_t(executor_->workers())};
    auto it = tunings_.find(key);
    if (it != tunings_.end()) return &it->second;
    if (tunings_.size() >= kMaxTunings) return nullptr;
    return &tunings_[key];
}

uint32_t DeviceCore::choose_width(uint32_t id, uint64_t function, uint64_t key, const std::vector<Trial>& trials,
                                  const std::vector<std::vector<uint64_t>>& slices) {
    std::vector<uint32_t> lanes;
    for (const Trial& tr : trials) lanes.push_back(tr.lanes);
    WidthTuning chosen;
    chosen.choose(lanes, slices);
    std::lock_guard lock(mutex_);
    auto it = contexts_.find(id);
    if (it == contexts_.end()) return chosen.chosen();
    auto fn = it->second.functions.find(function);
    if (fn == it->second.functions.end()) return chosen.chosen();
    WidthTuning* tp = tuning(fn->second, key);
    if (!tp) return chosen.chosen();
    WidthTuning& t = *tp;
    t = chosen;
    if (std::getenv("CPUSIM_VERBOSE")) {
        std::fprintf(stderr, "cpusim: width of %s, block %llux%llux%llu, in the first launch (us per 1000 blocks, estimate [range] of %d slices of %llu)",
                     fn->second.name.c_str(), (unsigned long long)(key >> 40), (unsigned long long)((key >> 20) & 0xfffff),
                     (unsigned long long)(key & 0xfffff), trial_rounds(), (unsigned long long)trial_blocks());
        std::fprintf(stderr, "%s -> %u%s lanes\n", t.describe().c_str(), t.chosen() & 0xffff, width_suffix(t.chosen()));
    }
    return t.chosen();
}

void DeviceCore::record_timing(uint32_t id, uint64_t function, uint64_t key, uint32_t lanes, uint64_t ns,
                               uint64_t work_ns) {
    std::lock_guard lock(mutex_);
    auto it = contexts_.find(id);
    if (it == contexts_.end()) return;
    auto fn = it->second.functions.find(function);
    if (fn == it->second.functions.end()) return;
    WidthTuning* tp = tuning(fn->second, key);
    if (!tp) return;
    WidthTuning& t = *tp;
    if (!t.record(lanes, ns, work_ns)) return;
    if (std::getenv("CPUSIM_VERBOSE")) {
        std::fprintf(stderr, "cpusim: width of %s, block %llux%llux%llu, over launches (us of critical path per 1000 blocks, estimate [range] of the runs; -: dropped)",
                     fn->second.name.c_str(), (unsigned long long)(key >> 40),
                     (unsigned long long)((key >> 20) & 0xfffff), (unsigned long long)(key & 0xfffff));
        std::fprintf(stderr, "%s -> %u%s lanes%s\n", t.describe().c_str(), t.chosen() & 0xffff, width_suffix(t.chosen()),
                     t.short_launches() ? " (short launches: the first)" : "");
    }
}

uint64_t DeviceCore::vram_resident() const {
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    {
        std::lock_guard lock(mutex_);
        ranges.assign(allocator_->allocations().begin(), allocator_->allocations().end());
    }
    return vram_->resident_bytes(ranges);
}

void DeviceCore::mem_info(uint64_t* free, uint64_t* total, uint64_t* largest_free) const {
    std::lock_guard lock(mutex_);
    *total = allocator_->size();
    *free = allocator_->size() - allocator_->used();
    if (largest_free) *largest_free = allocator_->largest_free();
}

}  // namespace cpusim
