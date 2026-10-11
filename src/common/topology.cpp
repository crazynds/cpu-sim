#include "common/topology.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>

#include "common/units.h"

namespace cpusim {

namespace {

std::string read_line(const std::string& path) {
    std::ifstream f(path);
    std::string line;
    std::getline(f, line);
    return line;
}

int read_int(const std::string& path, int fallback) {
    std::string s = read_line(path);
    try {
        return s.empty() ? fallback : std::stoi(s);
    } catch (...) {
        return fallback;
    }
}

}  // namespace

Topology read_topology() {
    Topology topo;
    auto online = parse_cpu_list(read_line("/sys/devices/system/cpu/online")).value_or(std::vector<int>{});
    topo.logical_cpus = static_cast<int>(online.size());

    std::map<std::pair<int, int>, PhysicalCore> by_id;
    for (int cpu : online) {
        std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        int pkg = read_int(base + "physical_package_id", 0);
        int core = read_int(base + "core_id", cpu);
        auto& pc = by_id[{pkg, core}];
        pc.package = pkg;
        pc.core_id = core;
        pc.cpus.push_back(cpu);
    }
    for (auto& [key, pc] : by_id) {
        std::sort(pc.cpus.begin(), pc.cpus.end());
        topo.cores.push_back(pc);
    }
    std::sort(topo.cores.begin(), topo.cores.end(),
              [](const PhysicalCore& a, const PhysicalCore& b) { return a.cpus.front() < b.cpus.front(); });

    std::ifstream info("/proc/cpuinfo");
    std::string line;
    while (std::getline(info, line)) {
        if (line.rfind("model name", 0) == 0) {
            auto colon = line.find(':');
            if (colon != std::string::npos) topo.cpu_model = line.substr(line.find_first_not_of(' ', colon + 1));
            break;
        }
    }
    if (topo.cpu_model.empty()) topo.cpu_model = "Unknown CPU";
    return topo;
}

std::string short_cpu_model(const std::string& model) {
    std::string s = model;
    for (const char* suffix : {" with Radeon Graphics", " Processor"}) {
        auto pos = s.find(suffix);
        if (pos != std::string::npos) s.erase(pos);
    }
    // "AMD Ryzen 9 7950X 16-Core" -> "AMD Ryzen 9 7950X"
    auto pos = s.find("-Core");
    if (pos != std::string::npos) {
        auto sp = s.rfind(' ', pos);
        if (sp != std::string::npos) s.erase(sp);
    }
    return s;
}

bool cpu_has_avx512() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl");
}

TensorLevel tensor_level() {
    __builtin_cpu_init();
    TensorLevel level = TensorLevel::Fp16;
    // vpdpbusd and the byte permutes (vpermb) of the int8 path; vpopcntd for
    // the 1-bit type, which compute capability 7.5 includes.
    if (__builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512vbmi") &&
        __builtin_cpu_supports("avx512vpopcntdq")) {
        level = TensorLevel::Int8;
        if (__builtin_cpu_supports("avx512bf16")) level = TensorLevel::Bf16;
    }
    if (const char* v = std::getenv("CPUSIM_TENSOR")) {
        TensorLevel want = level;
        if (std::strcmp(v, "fp16") == 0) want = TensorLevel::Fp16;
        else if (std::strcmp(v, "int8") == 0) want = TensorLevel::Int8;
        else if (std::strcmp(v, "bf16") == 0) want = TensorLevel::Bf16;
        if (want < level) level = want;
    }
    return level;
}

const char* tensor_level_isa(TensorLevel level) {
    switch (level) {
        case TensorLevel::Bf16: return "AVX512-BF16 + AVX512-VNNI";
        case TensorLevel::Int8: return "AVX512-VNNI + AVX512-VBMI + AVX512-VPOPCNTDQ";
        default: return "F16C + FMA";
    }
}

int compute_capability(TensorLevel level) {
    // The level a GPU first had these tensor core types at; with everything,
    // the architecture of the GPU this machine pairs with (RTX 50, sm_120).
    switch (level) {
        case TensorLevel::Bf16: return 120;
        case TensorLevel::Int8: return 75;
        default: return 70;
    }
}

}  // namespace cpusim
