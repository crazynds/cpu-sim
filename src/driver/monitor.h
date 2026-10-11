#pragma once

#include <condition_variable>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/sensors.h"
#include "device/device_core.h"

namespace cpusim {

// An application running the device in-process (CPUSIM_BACKEND=inprocess),
// as it published itself (see device/inprocess_status.h).
struct InProcessApp {
    int pid = 0;
    std::string name;
    int64_t started_unix = 0;
    uint64_t used_bytes = 0, allocations = 0;
    uint64_t kernels_launched = 0, kernels_profiled = 0;
    uint64_t bytes_htod = 0, bytes_dtoh = 0, bytes_dtod = 0, bytes_memset = 0;
    uint64_t alloc_calls = 0, free_calls = 0, failed_allocs = 0;
};

// What the monitor measured over its last window.
struct MonitorSample {
    double gpu_util = 0;     // % of the window with a kernel running (driver + in-process apps)
    double memory_util = 0;  // % of the window with a copy or memset running
    uint64_t vram_resident = 0;
    SensorReading sensors;
    std::vector<InProcessApp> inprocess;
};

// Samples utilization and host sensors once per second, like the GPU's own
// counters behind nvidia-smi.
class Monitor {
public:
    explicit Monitor(DeviceCore& core);
    ~Monitor();

    MonitorSample latest() const;

private:
    void run();
    // Reads the in-process apps' records; adds their busy ns since the last call.
    void scan_inprocess(MonitorSample* s, uint64_t* kernel_delta, uint64_t* copy_delta);

    DeviceCore& core_;
    SensorSampler sensors_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
    MonitorSample latest_;
    std::string inprocess_dir_;
    struct Busy {
        uint64_t kernel = 0, copy = 0;
    };
    std::map<std::string, Busy> prev_inprocess_;  // by record file name
    std::thread thread_;
};

}  // namespace cpusim
