#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace cpusim {

struct Config;

// An application on the in-process backend runs the device itself, so the
// driver never sees it. To still show it in cpu-smi, it publishes this record
// in a small shared file, <socket dir>/inprocess/<pid>, which the driver's
// monitor reads once per second (and removes when the process is gone).
constexpr uint64_t kInProcessStatusMagic = 0x3154535053504943ull;  // "CIPSPST1"

struct InProcessStatus {
    uint64_t magic;
    int32_t pid;
    int32_t reserved;
    int64_t started_unix;
    char name[256];
    // Busy time as a running integral: completed busy ns plus, while a kernel
    // runs, the time since it started (CLOCK_MONOTONIC ns, 0 when idle).
    std::atomic<uint64_t> kernel_busy_ns, kernel_since_ns, copy_busy_ns;
    std::atomic<uint64_t> used_bytes, allocations;
    std::atomic<uint64_t> kernels_launched, kernels_profiled;
    std::atomic<uint64_t> bytes_htod, bytes_dtoh, bytes_dtod, bytes_memset;
    std::atomic<uint64_t> alloc_calls, free_calls, failed_allocs;
};

std::string inprocess_status_dir(const Config& cfg);

// Busy ns up to now from a completed counter and a "running since" stamp.
uint64_t busy_now_ns(uint64_t completed, uint64_t since);
uint64_t monotonic_ns();

}  // namespace cpusim
