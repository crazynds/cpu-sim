#include "common/units.h"

#include <algorithm>
#include <cctype>
#include <charconv>

namespace cpusim {

namespace {

std::string_view trim(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
    return s;
}

std::optional<uint64_t> parse_u64(std::string_view s) {
    uint64_t v = 0;
    auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc() || p != s.data() + s.size()) return std::nullopt;
    return v;
}

}  // namespace

std::optional<uint64_t> parse_size(std::string_view text) {
    text = trim(text);
    size_t digits = 0;
    while (digits < text.size() && std::isdigit(static_cast<unsigned char>(text[digits]))) ++digits;
    if (digits == 0) return std::nullopt;
    auto value = parse_u64(text.substr(0, digits));
    if (!value) return std::nullopt;

    std::string unit;
    for (char c : trim(text.substr(digits))) unit += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    uint64_t mult = 1;
    if (unit.empty() || unit == "b") mult = 1;
    else if (unit == "k" || unit == "kb" || unit == "kib") mult = KiB;
    else if (unit == "m" || unit == "mb" || unit == "mib") mult = MiB;
    else if (unit == "g" || unit == "gb" || unit == "gib") mult = GiB;
    else if (unit == "t" || unit == "tb" || unit == "tib") mult = 1024 * GiB;
    else return std::nullopt;

    if (*value > UINT64_MAX / mult) return std::nullopt;
    return *value * mult;
}

std::string format_size(uint64_t bytes) {
    if (bytes != 0 && bytes % GiB == 0) return std::to_string(bytes / GiB) + "GiB";
    if (bytes != 0 && bytes % MiB == 0) return std::to_string(bytes / MiB) + "MiB";
    if (bytes != 0 && bytes % KiB == 0) return std::to_string(bytes / KiB) + "KiB";
    return std::to_string(bytes) + "B";
}

std::optional<std::vector<int>> parse_cpu_list(std::string_view text) {
    std::vector<int> cpus;
    text = trim(text);
    while (!text.empty()) {
        size_t comma = text.find(',');
        std::string_view item = trim(text.substr(0, comma));
        text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);
        if (item.empty()) return std::nullopt;

        size_t dash = item.find('-');
        auto lo = parse_u64(trim(item.substr(0, dash)));
        auto hi = dash == std::string_view::npos ? lo : parse_u64(trim(item.substr(dash + 1)));
        if (!lo || !hi || *hi < *lo || *hi > 65535) return std::nullopt;
        for (uint64_t c = *lo; c <= *hi; ++c) cpus.push_back(static_cast<int>(c));
    }
    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    return cpus;
}

std::string format_cpu_list(const std::vector<int>& cpus_in) {
    std::vector<int> cpus = cpus_in;
    std::sort(cpus.begin(), cpus.end());
    std::string out;
    for (size_t i = 0; i < cpus.size();) {
        size_t j = i;
        while (j + 1 < cpus.size() && cpus[j + 1] == cpus[j] + 1) ++j;
        if (!out.empty()) out += ',';
        out += std::to_string(cpus[i]);
        if (j > i) out += '-' + std::to_string(cpus[j]);
        i = j + 1;
    }
    return out;
}

}  // namespace cpusim
