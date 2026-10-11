#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace cpusim {

// Host sensors mapped onto what nvidia-smi shows. Negative values mean N/A.
struct SensorReading {
    double temperature_c = -1;    // k10temp Tctl / coretemp package
    double power_w = -1;          // RAPL package energy delta (usually root only)
    double clock_mhz = -1;        // mean current frequency of the reserved CPUs
    double max_clock_mhz = -1;
    double host_load = -1;        // % of time the reserved CPUs were busy (any process)
    std::vector<int> fan_rpm;
};

// Keeps the previous counters so each sample() reports rates over the interval.
class SensorSampler {
public:
    explicit SensorSampler(std::vector<int> cpus);
    SensorReading sample();

private:
    struct CpuTimes {
        uint64_t busy = 0, total = 0;
    };

    std::vector<int> cpus_;
    std::string temp_path_;
    std::vector<std::string> fan_paths_;
    std::string energy_path_;
    std::vector<CpuTimes> prev_cpu_;
    uint64_t prev_energy_uj_ = 0;
    bool have_energy_ = false;
    std::chrono::steady_clock::time_point prev_time_;
};

}  // namespace cpusim
