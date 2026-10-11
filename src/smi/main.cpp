// cpu-smi: nvidia-smi for the CPU SIMT device. Talks to the running driver.
#include <unistd.h>

#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <memory>
#include <map>
#include <string>
#include <vector>

#include "common/topology.h"
#include "common/units.h"
#include "device/driver_client.h"

using namespace cpusim;
using proto::DeviceStatus;

namespace {

constexpr int kExitNoDriver = 9;  // what nvidia-smi returns when the driver is not loaded

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return buf;
}

uint64_t mib(uint64_t bytes) { return (bytes + MiB / 2) / MiB; }

std::string uuid_string(const uint8_t u[16]) {
    std::string s = "CPU-";
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) s += '-';
        s += fmt("%02x", u[i]);
    }
    return s;
}

std::string now_string() {
    char buf[64];
    time_t t = time(nullptr);
    std::strftime(buf, sizeof(buf), "%a %b %e %H:%M:%S %Y", localtime(&t));
    return buf;
}

std::string time_string(int64_t unix) {
    char buf[64];
    time_t t = unix;
    std::strftime(buf, sizeof(buf), "%a %b %e %H:%M:%S %Y", localtime(&t));
    return buf;
}

std::string cuda_version(int v) { return fmt("%d.%d", v / 1000, (v % 1000) / 10); }

// Truncates like nvidia-smi: "NVIDIA GeForce RTX 5060 ..." for a column of `width`.
std::string fit(const std::string& s, size_t width) {
    if (s.size() <= width) return s;
    return s.substr(0, width - 3) + "...";
}

std::string or_na(double v, const char* f) { return v < 0 ? "N/A" : fmt(f, v); }

// ---------------------------------------------------------------- default view

// Pads or cuts `s` to exactly `width` columns.
std::string cell(const std::string& s, size_t width) {
    return s.size() >= width ? s.substr(0, width) : s + std::string(width - s.size(), ' ');
}

// One line of the device box: three cells, 91 columns in total, like nvidia-smi.
void box_row(const std::string& a, const std::string& b, const std::string& c) {
    std::printf("|%s|%s|%s|\n", cell(a, 41).c_str(), cell(b, 24).c_str(), cell(c, 22).c_str());
}
void full_row(const std::string& s) { std::printf("|%s|\n", cell(s, 89).c_str()); }

void print_table(const DeviceStatus& st) {
    const auto& p = st.props;
    const char* sep_full = "+-----------------------------------------------------------------------------------------+";
    const char* sep_box = "+-----------------------------------------+------------------------+----------------------+";
    std::printf("%s\n%s\n", now_string().c_str(), sep_full);
    full_row(fmt(" %-33s Driver Version: %-14s CUDA Version: %s", ("CPU-SMI " + std::string(st.driver_version)).c_str(),
                 st.driver_version, cuda_version(st.cuda_version).c_str()));
    std::printf("%s\n", sep_box);
    // Same layout as nvidia-smi. Dev-Util is the share of time a kernel runs on
    // the device (what GPU-Util means); CPU Load is how busy its cores really are.
    box_row(" CPU  Name                 Persistence-M", " Logical CPUs    Disp.A", " Volatile Uncorr. ECC");
    box_row(" Fan   Temp   Perf         Pwr:Usage/Cap", "           Memory-Usage", " Dev-Util  Compute M.");
    box_row("", "", "             CPU Load");
    std::printf("|=========================================+========================+======================|\n");

    std::string fan = st.fan_count ? fmt("%d", st.fan_rpm[0]) : "N/A";
    std::string temp = st.temperature_c < 0 ? "N/A" : fmt("%.0fC", st.temperature_c);
    std::string pwr = st.power_w < 0 ? "N/A" : fmt("%.0fW", st.power_w);
    box_row(fmt(" %3d  %-27s %5s  ", 0, fit(p.name, 27).c_str(), "On"),
            fmt(" %-18s %3s ", fit(std::string(st.reserved_cpus), 18).c_str(), "Off"), fmt("%21s ", "N/A"));
    box_row(fmt(" %4s  %4s   %4s  %13s / %4s ", fan.c_str(), temp.c_str(), fmt("P%d", st.pstate).c_str(), pwr.c_str(),
                "N/A"),
            fmt("%8luMiB / %6luMiB ", (unsigned long)mib(st.vram_used), (unsigned long)mib(st.vram_total)),
            fmt("%8s%13s ", fmt("%.0f%%", st.gpu_util).c_str(), "Default"));
    box_row("", "", fmt("%21s ", st.host_load < 0 ? "N/A" : fmt("%.0f%%", st.host_load).c_str()));
    std::printf("%s\n\n%s\n", sep_box, sep_full);
    full_row(" Processes:");
    full_row("  CPU   GI   CI              PID   Type   Process name                     Device Memory");
    full_row(fmt("        ID   ID%73s ", "Usage"));
    std::printf("|=========================================================================================|\n");
    if (st.process_count == 0) full_row("  No running processes found");
    bool any_inprocess = false;
    for (int i = 0; i < st.process_count; ++i) {
        const auto& pr = st.processes[i];
        std::string name = pr.name;
        if (name.size() > 35) name = "..." + name.substr(name.size() - 32);
        any_inprocess |= pr.inprocess != 0;
        full_row(fmt("  %3d   N/A  N/A  %15d      %s   %-35s %9s ", 0, pr.pid, pr.inprocess ? "I" : "C", name.c_str(),
                     fmt("%luMiB", (unsigned long)mib(pr.used_bytes)).c_str()));
    }
    if (any_inprocess) full_row("  Type I: runs the device in its own process (CPUSIM_BACKEND=inprocess)");
    std::printf("%s\n", sep_full);
}

// ---------------------------------------------------------------- -L

void print_list(const DeviceStatus& st) {
    std::printf("CPU 0: %s (UUID: %s)\n", st.props.name, uuid_string(st.props.uuid).c_str());
}

// ---------------------------------------------------------------- -q

void row(int indent, const std::string& key, const std::string& value) {
    std::string k(static_cast<size_t>(indent), ' ');
    k += key;
    std::printf("%-55s: %s\n", k.c_str(), value.c_str());
}
void section(int indent, const std::string& name) { std::printf("%*s%s\n", indent, "", name.c_str()); }

std::string mib_str(uint64_t bytes) { return fmt("%lu MiB", (unsigned long)mib(bytes)); }

void print_query(const DeviceStatus& st) {
    const auto& p = st.props;
    std::printf("\n==============CPUSMI LOG==============\n\n");
    row(0, "Timestamp", now_string());
    row(0, "Driver Version", st.driver_version);
    row(0, "CUDA Version", cuda_version(st.cuda_version));
    std::printf("\n");
    row(0, "Attached Devices", "1");
    std::printf("Device 0\n");
    row(4, "Product Name", p.name);
    row(4, "Product Architecture", "x86-64 AVX-512 (warp = 32 lanes)");
    row(4, "Host CPU", st.cpu_model);
    row(4, "Compute Capability", fmt("%d.%d", p.major, p.minor));
    {
        // The tensor core types that compute capability allows, and the
        // instructions that run them (common/topology.h).
        const int cc = p.major * 10 + p.minor;
        cpusim::TensorLevel level = cc >= 80 ? cpusim::TensorLevel::Bf16
                                    : cc >= 72 ? cpusim::TensorLevel::Int8
                                               : cpusim::TensorLevel::Fp16;
        row(4, "Tensor Cores", fmt("%s (%s)", cc >= 80 ? "fp16, int8, bf16" : cc >= 72 ? "fp16, int8" : "fp16",
                                   cpusim::tensor_level_isa(level)));
    }
    row(4, "Persistence Mode", "Enabled");
    row(4, "Addressing Mode", "UVA (device memory private to the driver)");
    row(4, "Device UUID", uuid_string(p.uuid));
    row(4, "Minor Number", "0");
    row(4, "Driver PID", std::to_string(st.driver_pid));
    row(4, "Driver Started", time_string(st.driver_started_unix));
    row(4, "Config", st.config_source);
    section(4, "Multiprocessors");
    row(8, "SM Count (physical cores)", std::to_string(st.reserved_cores));
    row(8, "Logical CPUs", st.reserved_cpus);
    row(8, "Host Cores / Logical CPUs", fmt("%d / %d", st.host_cores, st.host_logical_cpus));
    row(8, "Warp Size", std::to_string(p.warpSize));
    row(8, "Max Threads per Block", std::to_string(p.maxThreadsPerBlock));
    row(8, "Max Threads per SM", std::to_string(p.maxThreadsPerMultiProcessor));
    row(8, "Shared Memory per Block", fmt("%lu KiB", (unsigned long)(p.sharedMemPerBlock / KiB)));
    row(8, "Shared Memory per SM", fmt("%lu KiB", (unsigned long)(p.sharedMemPerMultiprocessor / KiB)));
    section(4, "FB Memory Usage");
    row(8, "Total", mib_str(st.vram_total));
    row(8, "Reserved", "0 MiB");
    row(8, "Used", mib_str(st.vram_used));
    row(8, "Free", mib_str(st.vram_free));
    row(8, "Largest Free Block", mib_str(st.vram_largest_free));
    row(8, "Resident in Host RAM", mib_str(st.vram_resident));
    row(8, "Base Address", fmt("0x%lx", (unsigned long)st.vram_base));
    row(8, "Page Size", st.hugepages);
    row(8, "Prefaulted", st.prefault ? "Yes" : "No");
    section(4, "Shared Memory Pool");
    row(8, "Total", mib_str(st.vram_shared_total));
    row(8, "Used", mib_str(st.vram_shared_used));
    row(4, "Compute Mode", "Default");
    section(4, "Utilization");
    row(8, "Device (time with a kernel running)", fmt("%.0f %%", st.gpu_util));
    row(8, "Memory (time copying)", fmt("%.0f %%", st.memory_util));
    row(8, "CPU Load on Reserved CPUs", or_na(st.host_load, "%.0f %%"));
    row(4, "Performance State", fmt("P%d", st.pstate));
    section(4, "Temperature");
    row(8, "CPU Current Temp", or_na(st.temperature_c, "%.0f C"));
    section(4, "Power Readings");
    row(8, "Power Draw", st.power_w < 0 ? "N/A (RAPL needs root)" : fmt("%.2f W", st.power_w));
    row(8, "Current Power Limit", "N/A");
    section(4, "Clocks");
    row(8, "SM", or_na(st.clock_mhz, "%.0f MHz"));
    section(4, "Max Clocks");
    row(8, "SM", or_na(st.max_clock_mhz, "%.0f MHz"));
    if (st.fan_count == 0) row(4, "Fan Speed", "N/A");
    for (int i = 0; i < st.fan_count; ++i) row(4, fmt("Fan %d Speed", i), fmt("%d RPM", st.fan_rpm[i]));
    section(4, "Statistics");
    row(8, "Kernels Launched", std::to_string(st.kernels_launched));
    row(8, "Kernels Profiled", std::to_string(st.kernels_profiled));
    row(8, "Allocations / Frees", fmt("%lu / %lu", (unsigned long)st.alloc_calls, (unsigned long)st.free_calls));
    row(8, "Failed Allocations", std::to_string(st.failed_allocs));
    row(8, "Host to Device", mib_str(st.bytes_htod));
    row(8, "Device to Host", mib_str(st.bytes_dtoh));
    row(8, "Device to Device", mib_str(st.bytes_dtod));
    row(8, "Memset", mib_str(st.bytes_memset));
    section(4, "Processes");
    if (st.process_count == 0) std::printf("%8sNone\n", "");
    for (int i = 0; i < st.process_count; ++i) {
        const auto& pr = st.processes[i];
        row(8, "Process ID", std::to_string(pr.pid));
        row(12, "Type", pr.inprocess ? "I" : "C");
        row(12, "Name", pr.name);
        row(12, "Backend", pr.inprocess ? "in-process" : "cpusimd");
        row(12, "Used Device Memory", mib_str(pr.used_bytes));
        row(12, "Allocations", std::to_string(pr.allocations));
        row(12, "Started", time_string(pr.started_unix));
    }
    std::printf("\n");
}

// ---------------------------------------------------------------- --query-*

struct CsvFormat {
    bool header = true, units = true;
};

using Getter = std::function<std::string(const DeviceStatus&, const CsvFormat&)>;

std::string with_unit(const CsvFormat& f, const std::string& v, const char* unit) {
    return f.units && v != "[N/A]" ? v + " " + unit : v;
}
std::string num_or_na(double v, const char* f) { return v < 0 ? "[N/A]" : fmt(f, v); }

const std::vector<std::pair<std::string, std::pair<std::string, Getter>>>& gpu_fields() {
    static const std::vector<std::pair<std::string, std::pair<std::string, Getter>>> fields = {
        {"index", {"index", [](auto&, auto&) { return std::string("0"); }}},
        {"name", {"name", [](auto& s, auto&) { return std::string(s.props.name); }}},
        {"uuid", {"uuid", [](auto& s, auto&) { return uuid_string(s.props.uuid); }}},
        {"driver_version", {"driver_version", [](auto& s, auto&) { return std::string(s.driver_version); }}},
        {"compute_cap", {"compute_cap", [](auto& s, auto&) { return fmt("%d.%d", s.props.major, s.props.minor); }}},
        {"count", {"count", [](auto&, auto&) { return std::string("1"); }}},
        {"pstate", {"pstate", [](auto& s, auto&) { return fmt("P%d", s.pstate); }}},
        {"memory.total", {"memory.total [MiB]", [](auto& s, auto& f) { return with_unit(f, std::to_string(mib(s.vram_total)), "MiB"); }}},
        {"memory.used", {"memory.used [MiB]", [](auto& s, auto& f) { return with_unit(f, std::to_string(mib(s.vram_used)), "MiB"); }}},
        {"memory.free", {"memory.free [MiB]", [](auto& s, auto& f) { return with_unit(f, std::to_string(mib(s.vram_free)), "MiB"); }}},
        {"utilization.gpu", {"utilization.gpu [%]", [](auto& s, auto& f) { return with_unit(f, fmt("%.0f", s.gpu_util), "%"); }}},
        {"utilization.memory", {"utilization.memory [%]", [](auto& s, auto& f) { return with_unit(f, fmt("%.0f", s.memory_util), "%"); }}},
        {"utilization.host", {"utilization.host [%]", [](auto& s, auto& f) { return with_unit(f, num_or_na(s.host_load, "%.0f"), "%"); }}},
        {"temperature.gpu", {"temperature.gpu", [](auto& s, auto&) { return num_or_na(s.temperature_c, "%.0f"); }}},
        {"power.draw", {"power.draw [W]", [](auto& s, auto& f) { return with_unit(f, num_or_na(s.power_w, "%.2f"), "W"); }}},
        {"clocks.sm", {"clocks.current.sm [MHz]", [](auto& s, auto& f) { return with_unit(f, num_or_na(s.clock_mhz, "%.0f"), "MHz"); }}},
        {"clocks.max.sm", {"clocks.max.sm [MHz]", [](auto& s, auto& f) { return with_unit(f, num_or_na(s.max_clock_mhz, "%.0f"), "MHz"); }}},
        {"fan.speed", {"fan.speed [RPM]", [](auto& s, auto& f) { return s.fan_count ? with_unit(f, std::to_string(s.fan_rpm[0]), "RPM") : std::string("[N/A]"); }}},
        {"multiprocessor_count", {"multiprocessor_count", [](auto& s, auto&) { return std::to_string(s.props.multiProcessorCount); }}},
        {"cpus", {"cpus", [](auto& s, auto&) { return std::string(s.reserved_cpus); }}},
    };
    return fields;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t end = s.find(sep, start);
        if (end == std::string::npos) end = s.size();
        std::string item = s.substr(start, end - start);
        while (!item.empty() && item.front() == ' ') item.erase(0, 1);
        while (!item.empty() && item.back() == ' ') item.pop_back();
        if (!item.empty()) out.push_back(item);
        start = end + 1;
    }
    return out;
}

bool print_query_gpu(const DeviceStatus& st, const std::string& list, const CsvFormat& f) {
    std::vector<const std::pair<std::string, Getter>*> cols;
    for (const auto& name : split(list, ',')) {
        const std::pair<std::string, Getter>* found = nullptr;
        for (const auto& [key, field] : gpu_fields())
            if (key == name) found = &field;
        if (!found) {
            std::fprintf(stderr, "Field \"%s\" is not a valid field to query.\n", name.c_str());
            return false;
        }
        cols.push_back(found);
    }
    for (int pass = f.header ? 0 : 1; pass < 2; ++pass) {
        for (size_t i = 0; i < cols.size(); ++i)
            std::printf("%s%s", i ? ", " : "", pass == 0 ? cols[i]->first.c_str() : cols[i]->second(st, f).c_str());
        std::printf("\n");
    }
    return true;
}

bool print_query_apps(const DeviceStatus& st, const std::string& list, const CsvFormat& f) {
    auto names = split(list, ',');
    for (const auto& n : names) {
        if (n != "pid" && n != "process_name" && n != "used_memory" && n != "gpu_uuid" && n != "gpu_name") {
            std::fprintf(stderr, "Field \"%s\" is not a valid field to query.\n", n.c_str());
            return false;
        }
    }
    if (f.header) {
        for (size_t i = 0; i < names.size(); ++i)
            std::printf("%s%s", i ? ", " : "", names[i] == "used_memory" ? "used_gpu_memory [MiB]" : names[i].c_str());
        std::printf("\n");
    }
    for (int p = 0; p < st.process_count; ++p) {
        const auto& pr = st.processes[p];
        for (size_t i = 0; i < names.size(); ++i) {
            std::string v;
            if (names[i] == "pid") v = std::to_string(pr.pid);
            else if (names[i] == "process_name") v = pr.name;
            else if (names[i] == "used_memory") v = with_unit(f, std::to_string(mib(pr.used_bytes)), "MiB");
            else if (names[i] == "gpu_uuid") v = uuid_string(st.props.uuid);
            else v = st.props.name;
            std::printf("%s%s", i ? ", " : "", v.c_str());
        }
        std::printf("\n");
    }
    return true;
}

void usage() {
    std::printf(
        "CPU-SMI %s: System Management Interface for the CPU SIMT device.\n"
        "\n"
        "usage: cpu-smi [options]\n"
        "\n"
        "  (no options)                 summary table, like nvidia-smi\n"
        "  -L, --list-gpus              list devices with UUIDs\n"
        "  -q, --query                  detailed report\n"
        "  -l, --loop[=SEC]             repeat every SEC seconds (default 1)\n"
        "  --query-gpu=FIELDS           CSV of device fields, e.g. memory.used,utilization.gpu\n"
        "  --query-compute-apps=FIELDS  CSV of processes: pid,process_name,used_memory\n"
        "  --format=csv[,noheader][,nounits]\n"
        "  --help-query-gpu             list the fields --query-gpu accepts\n"
        "  -h, --help\n",
        CPUSIM_VERSION);
}

}  // namespace

int main(int argc, char** argv) {
    enum class Mode { Table, List, Query, QueryGpu, QueryApps } mode = Mode::Table;
    std::string fields;
    CsvFormat csv;
    double loop = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto value_of = [&](const std::string& prefix) { return a.substr(prefix.size()); };
        if (a == "-h" || a == "--help") return usage(), 0;
        else if (a == "-L" || a == "--list-gpus") mode = Mode::List;
        else if (a == "-q" || a == "--query") mode = Mode::Query;
        else if (a == "-l" || a == "--loop") {
            loop = 1;
            if (i + 1 < argc && std::isdigit(static_cast<unsigned char>(argv[i + 1][0]))) loop = std::atof(argv[++i]);
        } else if (a.rfind("--loop=", 0) == 0) loop = std::atof(value_of("--loop=").c_str());
        else if (a.rfind("-l", 0) == 0 && a.size() > 2) loop = std::atof(a.c_str() + 2);
        else if (a.rfind("--query-gpu=", 0) == 0) mode = Mode::QueryGpu, fields = value_of("--query-gpu=");
        else if (a.rfind("--query-compute-apps=", 0) == 0) mode = Mode::QueryApps, fields = value_of("--query-compute-apps=");
        else if (a.rfind("--format=", 0) == 0) {
            for (const auto& opt : split(value_of("--format="), ',')) {
                if (opt == "noheader") csv.header = false;
                else if (opt == "nounits") csv.units = false;
                else if (opt != "csv") return std::fprintf(stderr, "Invalid format '%s'.\n", opt.c_str()), 2;
            }
        } else if (a == "--help-query-gpu") {
            for (const auto& [key, field] : gpu_fields()) std::printf("%s\n", key.c_str());
            return 0;
        } else {
            std::fprintf(stderr, "Invalid combination of input arguments: %s\n\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (loop < 0) loop = 0;

    auto st = std::make_unique<DeviceStatus>();
    while (true) {
        std::string message;
        Error e = query_driver_status(st.get(), &message);
        if (e != Error::Success) {
            std::fprintf(stderr,
                         "CPU-SMI has failed because it couldn't communicate with the cpusim driver. "
                         "Make sure that cpusimd is running.\n(%s)\n",
                         message.c_str());
            return kExitNoDriver;
        }
        bool ok = true;
        switch (mode) {
            case Mode::Table:
                if (loop > 0 && isatty(1)) std::printf("\033[H\033[2J");
                print_table(*st);
                break;
            case Mode::List: print_list(*st); break;
            case Mode::Query: print_query(*st); break;
            case Mode::QueryGpu: ok = print_query_gpu(*st, fields, csv); break;
            case Mode::QueryApps: ok = print_query_apps(*st, fields, csv); break;
        }
        std::fflush(stdout);
        if (!ok) return 2;
        if (loop <= 0) return 0;
        usleep(static_cast<useconds_t>(loop * 1e6));
    }
}
