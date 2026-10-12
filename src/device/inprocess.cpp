#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <condition_variable>
#include <cstring>
#include <ctime>
#include <mutex>
#include <thread>

#include "device/device.h"
#include "device/device_core.h"
#include "device/inprocess_status.h"

namespace cpusim {

namespace {

std::string self_exe() {
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    return n > 0 ? std::string(buf, size_t(n)) : std::string();
}

// Publishes this process's device activity for the driver's monitor (see
// inprocess_status.h), four times per second. Best effort: if the directory
// cannot be created the program simply does not show up in cpu-smi.
class StatusPublisher {
public:
    StatusPublisher(DeviceCore& core, uint32_t ctx) : core_(core), ctx_(ctx) {
        std::string dir = inprocess_status_dir(core.config());
        mkdir(dir.substr(0, dir.rfind('/')).c_str(), 0700);
        mkdir(dir.c_str(), 0700);
        path_ = dir + "/" + std::to_string(getpid());
        int fd = open(path_.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (fd < 0) return;
        if (ftruncate(fd, sizeof(InProcessStatus)) == 0) {
            void* p = mmap(nullptr, sizeof(InProcessStatus), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (p != MAP_FAILED) st_ = static_cast<InProcessStatus*>(p);
        }
        close(fd);
        if (!st_) {
            unlink(path_.c_str());
            return;
        }
        st_->pid = getpid();
        st_->started_unix = time(nullptr);
        std::snprintf(st_->name, sizeof(st_->name), "%s", self_exe().c_str());
        publish();
        st_->magic = kInProcessStatusMagic;
        thread_ = std::thread([this] {
            std::unique_lock lock(mutex_);
            while (!cv_.wait_for(lock, std::chrono::milliseconds(250), [this] { return stop_; })) publish();
        });
    }
    ~StatusPublisher() {
        if (!st_) return;
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        thread_.join();
        unlink(path_.c_str());
        munmap(st_, sizeof(InProcessStatus));
    }

private:
    void publish() {
        const auto& c = core_.counters();
        auto copy = [](std::atomic<uint64_t>& to, const std::atomic<uint64_t>& from) {
            to.store(from.load(std::memory_order_relaxed), std::memory_order_relaxed);
        };
        copy(st_->kernel_busy_ns, c.kernel_busy_ns);
        copy(st_->kernel_since_ns, c.kernel_since_ns);
        copy(st_->copy_busy_ns, c.copy_busy_ns);
        copy(st_->kernels_launched, c.kernels_launched);
        copy(st_->kernels_profiled, c.kernels_profiled);
        copy(st_->bytes_htod, c.bytes_htod);
        copy(st_->bytes_dtoh, c.bytes_dtoh);
        copy(st_->bytes_dtod, c.bytes_dtod);
        copy(st_->bytes_memset, c.bytes_memset);
        copy(st_->alloc_calls, c.alloc_calls);
        copy(st_->free_calls, c.free_calls);
        copy(st_->failed_allocs, c.failed_allocs);
        for (const auto& ctx : core_.contexts())
            if (ctx.id == ctx_) {
                st_->used_bytes.store(ctx.used_bytes, std::memory_order_relaxed);
                st_->allocations.store(ctx.allocations, std::memory_order_relaxed);
            }
    }

    DeviceCore& core_;
    uint32_t ctx_;
    std::string path_;
    InProcessStatus* st_ = nullptr;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
    std::thread thread_;
};

// The application owns the device: VRAM lives in this process and copies are
// plain memcpy. Same DeviceCore as the driver, so both backends share the
// allocator and the validation rules.
class InProcessDevice final : public Device {
public:
    explicit InProcessDevice(std::unique_ptr<DeviceCore> core)
        : core_(std::move(core)), ctx_(core_->open_context(getpid(), self_exe())), status_(*core_, ctx_) {}
    ~InProcessDevice() override { core_->close_context(ctx_); }

    const char* backend_name() const override { return "inprocess"; }
    const DeviceProperties& properties() const override { return core_->properties(); }
    std::string driver_version() const override { return CPUSIM_VERSION; }

    Error malloc(uint64_t bytes, uint64_t* dptr) override { return core_->allocate(ctx_, bytes, dptr); }
    Error free(uint64_t dptr) override { return core_->release(ctx_, dptr); }
    Error malloc_managed(uint64_t bytes, uint64_t* dptr) override { return core_->allocate_managed(ctx_, bytes, dptr); }
    bool is_managed_pointer(const void* ptr) const override {
        auto p = reinterpret_cast<const uint8_t*>(ptr);
        return core_->shared_base() && p >= core_->shared_base() && p < core_->shared_base() + core_->shared_size();
    }

    Error memcpy_htod(uint64_t dst, const void* src, uint64_t n) override {
        if (n == 0) return Error::Success;
        uint8_t* d = nullptr;
        if (Error e = core_->translate(ctx_, dst, n, &d); e != Error::Success) return e;
        BusyScope busy(core_->counters().copy_busy_ns);
        std::memcpy(d, src, n);
        core_->counters().bytes_htod.fetch_add(n, std::memory_order_relaxed);
        return Error::Success;
    }

    Error memcpy_dtoh(void* dst, uint64_t src, uint64_t n) override {
        if (n == 0) return Error::Success;
        uint8_t* s = nullptr;
        if (Error e = core_->translate(ctx_, src, n, &s); e != Error::Success) return e;
        BusyScope busy(core_->counters().copy_busy_ns);
        std::memcpy(dst, s, n);
        core_->counters().bytes_dtoh.fetch_add(n, std::memory_order_relaxed);
        return Error::Success;
    }

    Error memcpy_dtod(uint64_t dst, uint64_t src, uint64_t n) override { return core_->memcpy_dtod(ctx_, dst, src, n); }
    Error memset(uint64_t dst, int value, uint64_t n) override { return core_->memset(ctx_, dst, value, n); }

    Error mem_get_info(uint64_t* free, uint64_t* total) override {
        core_->mem_info(free, total);
        return Error::Success;
    }

    Error synchronize() override { return Error::Success; }

    Error load_module(const void* image, size_t size, uint64_t* module) override {
        int fd = make_memfd("cpusim-module", image, size);
        if (fd < 0) return Error::OperatingSystem;
        Error e = core_->load_module(ctx_, fd, module);
        close(fd);
        return e;
    }
    Error get_function(uint64_t module, const std::string& name, uint64_t* function) override {
        return core_->get_function(ctx_, module, name, function);
    }
    Error bind_symbols(uint64_t module, const std::string& table, bool init) override {
        return core_->bind_symbols(ctx_, module, table, init);
    }
    Error launch(uint64_t function, const LaunchConfig& cfg, const void* params, size_t param_bytes,
                 LaunchResult* result) override {
        return core_->launch(ctx_, function, cfg, params, param_bytes, result);
    }

    bool is_device_pointer(const void* ptr) const override {
        auto p = reinterpret_cast<const uint8_t*>(ptr);
        return p >= core_->vram().base() && p < core_->vram().base() + core_->vram().size();
    }

private:
    std::unique_ptr<DeviceCore> core_;
    uint32_t ctx_;
    StatusPublisher status_;
};

}  // namespace

std::unique_ptr<Device> open_inprocess_device(Error* error, std::string* message) {
    std::string err;
    auto cfg = load_config(&err);
    if (!cfg) {
        *error = Error::InitializationError;
        *message = err;
        return nullptr;
    }
    auto core = DeviceCore::create(*cfg, 0, &err);
    if (!core) {
        *error = Error::NoDevice;
        *message = err;
        return nullptr;
    }
    *error = Error::Success;
    return std::make_unique<InProcessDevice>(std::move(core));
}

}  // namespace cpusim
