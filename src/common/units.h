#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cpusim {

constexpr uint64_t KiB = 1024;
constexpr uint64_t MiB = 1024 * KiB;
constexpr uint64_t GiB = 1024 * MiB;

// Parses "16GiB", "512MiB", "2MB", "4096" (bytes). Decimal suffixes (KB, MB, GB)
// are treated as binary, the same way nvidia-smi reports memory.
std::optional<uint64_t> parse_size(std::string_view text);

// "16GiB", "512MiB" or "123B": the largest unit that divides the value exactly.
std::string format_size(uint64_t bytes);

// Parses a Linux CPU list such as "0-3,8,10-11".
std::optional<std::vector<int>> parse_cpu_list(std::string_view text);
std::string format_cpu_list(const std::vector<int>& cpus);

}  // namespace cpusim
