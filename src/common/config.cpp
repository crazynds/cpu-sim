#include "common/config.h"

#include <cctype>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "common/paths.h"
#include "common/units.h"

namespace cpusim {

const char* hugepages_name(HugePages h) {
    switch (h) {
        case HugePages::None: return "none";
        case HugePages::TwoMiB: return "2MiB";
        case HugePages::OneGiB: return "1GiB";
    }
    return "none";
}

namespace {

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

// Strips a trailing comment, ignoring '#' inside a quoted string.
std::string strip_comment(const std::string& line) {
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '"') quoted = !quoted;
        else if (line[i] == '#' && !quoted) return line.substr(0, i);
    }
    return line;
}

struct Value {
    enum Kind { String, Integer, Bool } kind;
    std::string str;
    int64_t integer = 0;
    bool boolean = false;
};

std::optional<Value> parse_value(const std::string& raw) {
    if (raw.size() >= 2 && raw.front() == '"' && raw.back() == '"')
        return Value{Value::String, raw.substr(1, raw.size() - 2)};
    if (raw == "true" || raw == "false") {
        Value v{Value::Bool, {}};
        v.boolean = raw == "true";
        return v;
    }
    int64_t n = 0;
    auto [p, ec] = std::from_chars(raw.data(), raw.data() + raw.size(), n);
    if (ec == std::errc() && p == raw.data() + raw.size()) {
        Value v{Value::Integer, {}};
        v.integer = n;
        return v;
    }
    return std::nullopt;
}

bool apply(Config& cfg, const std::string& section, const std::string& key, const Value& v, std::string* err) {
    auto want = [&](Value::Kind k, const char* what) {
        if (v.kind == k) return true;
        *err = "'" + key + "' expects " + what;
        return false;
    };
    auto size_value = [&](uint64_t& out) {
        if (!want(Value::String, "a size string such as \"8GiB\"")) return false;
        auto s = parse_size(v.str);
        if (!s) {
            *err = "invalid size '" + v.str + "'";
            return false;
        }
        out = *s;
        return true;
    };

    if (section == "device") {
        if (key == "name") return want(Value::String, "a string") && (cfg.name = v.str, true);
        if (key == "cores") {
            if (!want(Value::Integer, "an integer")) return false;
            if (v.integer < 1 || v.integer > 4096) return *err = "'cores' must be between 1 and 4096", false;
            cfg.cores = static_cast<int>(v.integer);
            return true;
        }
        if (key == "cpus") {
            if (!want(Value::String, "a CPU list string such as \"4-15,20-31\"")) return false;
            auto list = parse_cpu_list(v.str);
            if (!list || list->empty()) return *err = "invalid CPU list '" + v.str + "'", false;
            cfg.cpus = *list;
            return true;
        }
        if (key == "vram_private") return size_value(cfg.vram_private);
        if (key == "vram_shared") return size_value(cfg.vram_shared);
        if (key == "hugepages") {
            if (!want(Value::String, "\"2MiB\", \"1GiB\" or \"none\"")) return false;
            if (v.str == "2MiB") cfg.hugepages = HugePages::TwoMiB;
            else if (v.str == "1GiB") cfg.hugepages = HugePages::OneGiB;
            else if (v.str == "none") cfg.hugepages = HugePages::None;
            else return *err = "'hugepages' must be \"2MiB\", \"1GiB\" or \"none\"", false;
            return true;
        }
        if (key == "prefault") return want(Value::Bool, "true or false") && (cfg.prefault = v.boolean, true);
    } else if (section == "driver") {
        if (key == "socket") return want(Value::String, "a path string") && (cfg.socket = v.str, true);
    } else {
        *err = "unknown section [" + section + "]";
        return false;
    }
    *err = "unknown key '" + key + "' in [" + section + "]";
    return false;
}

}  // namespace

std::optional<Config> parse_config(const std::string& text, std::string* error) {
    Config cfg;
    std::string section;
    std::istringstream in(text);
    std::string line;
    int lineno = 0;
    std::string err;
    while (std::getline(in, line)) {
        ++lineno;
        line = trim(strip_comment(line));
        if (line.empty()) continue;
        if (line.front() == '[') {
            if (line.back() != ']') {
                err = "malformed section header";
            } else {
                section = trim(line.substr(1, line.size() - 2));
                if (section != "device" && section != "driver") err = "unknown section [" + section + "]";
            }
        } else {
            size_t eq = line.find('=');
            if (eq == std::string::npos) {
                err = "expected 'key = value'";
            } else if (section.empty()) {
                err = "key outside of a section";
            } else {
                std::string key = trim(line.substr(0, eq));
                auto value = parse_value(trim(line.substr(eq + 1)));
                if (!value) err = "invalid value for '" + key + "'";
                else apply(cfg, section, key, *value, &err);
            }
        }
        if (!err.empty()) {
            if (error) *error = "line " + std::to_string(lineno) + ": " + err;
            return std::nullopt;
        }
    }
    return cfg;
}

std::optional<Config> load_config(std::string* error) {
    std::string path = config_path();
    Config cfg;
    std::ifstream f(path);
    if (f) {
        std::stringstream buf;
        buf << f.rdbuf();
        std::string err;
        auto parsed = parse_config(buf.str(), &err);
        if (!parsed) {
            if (error) *error = path + ": " + err;
            return std::nullopt;
        }
        cfg = *parsed;
        cfg.source = path;
    } else {
        cfg.source = "(defaults)";
    }

    if (const char* v = std::getenv("CPUSIM_VRAM")) {
        auto s = parse_size(v);
        if (!s) {
            if (error) *error = std::string("invalid CPUSIM_VRAM '") + v + "'";
            return std::nullopt;
        }
        cfg.vram_private = *s;
    }
    if (const char* v = std::getenv("CPUSIM_CORES")) {
        int n = std::atoi(v);
        if (n < 1) {
            if (error) *error = std::string("invalid CPUSIM_CORES '") + v + "'";
            return std::nullopt;
        }
        cfg.cores = n;
        cfg.cpus.clear();
    }
    if (const char* v = std::getenv("CPUSIM_SOCKET")) cfg.socket = v;
    return cfg;
}

}  // namespace cpusim
