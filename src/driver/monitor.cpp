#include "driver/monitor.h"

#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>

#include "device/inprocess_status.h"

namespace cpusim {

Monitor::Monitor(DeviceCore& core)
    : core_(core), sensors_(core.reserved_cpus()), inprocess_dir_(inprocess_status_dir(core.config())) {
    latest_.sensors = sensors_.sample();
    latest_.vram_resident = core_.vram_resident();
    uint64_t k = 0, c = 0;
    scan_inprocess(&latest_, &k, &c);  // primes the per-app deltas
    thread_ = std::thread([this] { run(); });
}

Monitor::~Monitor() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
}

MonitorSample Monitor::latest() const {
    std::lock_guard lock(mutex_);
    return latest_;
}

void Monitor::scan_inprocess(MonitorSample* s, uint64_t* kernel_delta, uint64_t* copy_delta) {
    std::map<std::string, Busy> seen;
    DIR* d = opendir(inprocess_dir_.c_str());
    if (!d) {
        prev_inprocess_.clear();
        return;
    }
    while (dirent* e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        std::string path = inprocess_dir_ + "/" + e->d_name;
        int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) continue;
        void* p = mmap(nullptr, sizeof(InProcessStatus), PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (p == MAP_FAILED) continue;
        const auto* st = static_cast<const InProcessStatus*>(p);
        if (st->magic == kInProcessStatusMagic) {
            if (kill(st->pid, 0) != 0 && errno == ESRCH) {
                unlink(path.c_str());  // the process died without cleaning up
            } else {
                auto ld = [](const std::atomic<uint64_t>& a) { return a.load(std::memory_order_relaxed); };
                InProcessApp app;
                app.pid = st->pid;
                app.name.assign(st->name, strnlen(st->name, sizeof(st->name)));
                app.started_unix = st->started_unix;
                app.used_bytes = ld(st->used_bytes);
                app.allocations = ld(st->allocations);
                app.kernels_launched = ld(st->kernels_launched);
                app.kernels_profiled = ld(st->kernels_profiled);
                app.bytes_htod = ld(st->bytes_htod);
                app.bytes_dtoh = ld(st->bytes_dtoh);
                app.bytes_dtod = ld(st->bytes_dtod);
                app.bytes_memset = ld(st->bytes_memset);
                app.alloc_calls = ld(st->alloc_calls);
                app.free_calls = ld(st->free_calls);
                app.failed_allocs = ld(st->failed_allocs);
                s->inprocess.push_back(std::move(app));

                Busy now{busy_now_ns(ld(st->kernel_busy_ns), ld(st->kernel_since_ns)), ld(st->copy_busy_ns)};
                if (auto it = prev_inprocess_.find(e->d_name); it != prev_inprocess_.end()) {
                    *kernel_delta += now.kernel > it->second.kernel ? now.kernel - it->second.kernel : 0;
                    *copy_delta += now.copy > it->second.copy ? now.copy - it->second.copy : 0;
                }
                seen[e->d_name] = now;
            }
        }
        munmap(p, sizeof(InProcessStatus));
    }
    closedir(d);
    prev_inprocess_ = std::move(seen);
}

void Monitor::run() {
    using clock = std::chrono::steady_clock;
    auto& counters = core_.counters();
    auto kernel_now = [&] { return busy_now_ns(counters.kernel_busy_ns.load(), counters.kernel_since_ns.load()); };
    uint64_t prev_kernel = kernel_now();
    uint64_t prev_copy = counters.copy_busy_ns.load();
    auto prev_time = clock::now();

    std::unique_lock lock(mutex_);
    while (!cv_.wait_for(lock, std::chrono::seconds(1), [this] { return stop_; })) {
        lock.unlock();
        auto now = clock::now();
        double window_ns = std::chrono::duration<double, std::nano>(now - prev_time).count();
        uint64_t kernel = kernel_now();
        uint64_t copy = counters.copy_busy_ns.load();

        MonitorSample s;
        uint64_t kernel_delta = kernel > prev_kernel ? kernel - prev_kernel : 0;
        uint64_t copy_delta = copy - prev_copy;
        scan_inprocess(&s, &kernel_delta, &copy_delta);
        s.gpu_util = std::clamp(100.0 * double(kernel_delta) / window_ns, 0.0, 100.0);
        s.memory_util = std::clamp(100.0 * double(copy_delta) / window_ns, 0.0, 100.0);
        s.sensors = sensors_.sample();
        s.vram_resident = core_.vram_resident();

        prev_kernel = kernel;
        prev_copy = copy;
        prev_time = now;
        lock.lock();
        latest_ = s;
    }
}

}  // namespace cpusim
