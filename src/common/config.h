#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cpusim {

enum class HugePages { None, TwoMiB, OneGiB };

const char* hugepages_name(HugePages h);

// Contents of device.toml. Zero/empty fields mean "pick a default when the
// device is created" (see DeviceCore), so the file can stay short.
struct Config {
    // [device]
    std::string name;             // empty: derived from the CPU model
    int cores = 0;                // physical cores to reserve ("SMs"); 0 = default
    std::vector<int> cpus;        // explicit logical CPUs; expanded to whole physical cores
    uint64_t vram_private = 0;    // 0 = default
    // Shared pool (cudaMallocManaged, mapped host memory): a memory file mapped
    // at the same address by the driver and every application, so a pointer
    // into it works on both sides. Reserved address space; RAM is used only as
    // it is touched. 0 disables it.
    uint64_t vram_shared = 4ull << 30;
    HugePages hugepages = HugePages::TwoMiB;
    bool prefault = false;        // touch all of the VRAM at startup and never give it back

    // [driver]
    std::string socket;           // empty: $XDG_RUNTIME_DIR/cpusim/driver.sock

    std::string source;           // file it was loaded from, or "(defaults)"
};

// Parses the TOML subset we use: [sections], key = "string" | integer | true/false,
// comments with '#'. Unknown keys are errors, so typos do not go unnoticed.
std::optional<Config> parse_config(const std::string& text, std::string* error);

// Loads $CPUSIM_CONFIG, or ~/.config/cpusim/device.toml, or defaults when the
// file does not exist. Environment overrides (handy for tests and quick runs):
// CPUSIM_VRAM, CPUSIM_CORES, CPUSIM_SOCKET.
std::optional<Config> load_config(std::string* error);

}  // namespace cpusim
