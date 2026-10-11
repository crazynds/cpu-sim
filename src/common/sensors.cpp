#include "common/sensors.h"

#include <dirent.h>
#include <unistd.h>

#include <fstream>
#include <sstream>

namespace cpusim {

namespace {

std::string read_line(const std::string& path) {
    std::ifstream f(path);
    std::string line;
    std::getline(f, line);
    return line;
}

bool read_u64(const std::string& path, uint64_t* out) {
    std::ifstream f(path);
    return static_cast<bool>(f >> *out);
}

std::vector<std::string> list_dir(const std::string& path) {
    std::vector<std::string> names;
    if (DIR* d = opendir(path.c_str())) {
        while (dirent* e = readdir(d))
            if (e->d_name[0] != '.') names.push_back(e->d_name);
        closedir(d);
    }
    return names;
}

}  // namespace

SensorSampler::SensorSampler(std::vector<int> cpus) : cpus_(std::move(cpus)) {
    const std::string hwmon = "/sys/class/hwmon/";
    std::vector<std::string> acpi_fans;
    for (const auto& dev : list_dir(hwmon)) {
        std::string base = hwmon + dev + "/";
        std::string name = read_line(base + "name");
        if (temp_path_.empty() && (name == "k10temp" || name == "coretemp" || name == "zenpower")) {
            // Prefer the control temperature (Tctl) or the package sensor.
            std::string fallback;
            for (int i = 1; i <= 16; ++i) {
                std::string input = base + "temp" + std::to_string(i) + "_input";
                if (access(input.c_str(), R_OK) != 0) continue;
                std::string label = read_line(base + "temp" + std::to_string(i) + "_label");
                if (label == "Tctl" || label == "Tdie" || label.rfind("Package", 0) == 0) {
                    temp_path_ = input;
                    break;
                }
                if (fallback.empty()) fallback = input;
            }
            if (temp_path_.empty()) temp_path_ = fallback;
        }
        for (int i = 1; i <= 8; ++i) {
            std::string input = base + "fan" + std::to_string(i) + "_input";
            if (access(input.c_str(), R_OK) == 0) (name == "acpi_fan" ? acpi_fans : fan_paths_).push_back(input);
        }
    }
    // The generic ACPI fan device repeats a fan the vendor driver (hp, thinkpad,
    // asus, dell...) already reports: use it only when there is nothing better.
    if (fan_paths_.empty()) fan_paths_ = acpi_fans;

    for (const char* p : {"/sys/class/powercap/intel-rapl:0/energy_uj", "/sys/class/powercap/amd-rapl:0/energy_uj"}) {
        if (access(p, R_OK) == 0) {
            energy_path_ = p;
            break;
        }
    }
    prev_cpu_.resize(cpus_.size());
    sample();  // prime the deltas
}

SensorReading SensorSampler::sample() {
    SensorReading r;
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - prev_time_).count();

    uint64_t v = 0;
    if (!temp_path_.empty() && read_u64(temp_path_, &v)) r.temperature_c = v / 1000.0;
    for (const auto& p : fan_paths_)
        if (read_u64(p, &v)) r.fan_rpm.push_back(static_cast<int>(v));

    if (!energy_path_.empty() && read_u64(energy_path_, &v)) {
        if (have_energy_ && v >= prev_energy_uj_ && dt > 0) r.power_w = (v - prev_energy_uj_) / 1e6 / dt;
        prev_energy_uj_ = v;
        have_energy_ = true;
    }

    double sum_khz = 0;
    int n_khz = 0;
    for (int cpu : cpus_) {
        std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/";
        if (read_u64(base + "scaling_cur_freq", &v)) {
            sum_khz += v;
            ++n_khz;
        }
        if (r.max_clock_mhz < 0 && read_u64(base + "cpuinfo_max_freq", &v)) r.max_clock_mhz = v / 1000.0;
    }
    if (n_khz > 0) r.clock_mhz = sum_khz / n_khz / 1000.0;

    // /proc/stat: cpuN user nice system idle iowait irq softirq steal ...
    std::ifstream stat("/proc/stat");
    std::string line;
    uint64_t busy_delta = 0, total_delta = 0;
    while (std::getline(stat, line)) {
        if (line.rfind("cpu", 0) != 0 || line.size() < 4 || !isdigit(static_cast<unsigned char>(line[3]))) continue;
        std::istringstream ls(line.substr(3));
        int cpu = -1;
        ls >> cpu;
        for (size_t i = 0; i < cpus_.size(); ++i) {
            if (cpus_[i] != cpu) continue;
            uint64_t f[8] = {};
            for (auto& x : f) ls >> x;
            uint64_t idle = f[3] + f[4];
            uint64_t total = 0;
            for (auto x : f) total += x;
            CpuTimes cur{total - idle, total};
            if (prev_cpu_[i].total != 0) {
                busy_delta += cur.busy - prev_cpu_[i].busy;
                total_delta += cur.total - prev_cpu_[i].total;
            }
            prev_cpu_[i] = cur;
        }
    }
    if (total_delta > 0) r.host_load = 100.0 * busy_delta / total_delta;

    prev_time_ = now;
    return r;
}

}  // namespace cpusim
