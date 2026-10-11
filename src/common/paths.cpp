#include "common/paths.h"

#include <pwd.h>
#include <unistd.h>

#include <cstdlib>

#include "common/config.h"

namespace cpusim {

namespace {

std::string home_dir() {
    if (const char* h = std::getenv("HOME"); h && *h) return h;
    if (passwd* pw = getpwuid(getuid())) return pw->pw_dir;
    return "/tmp";
}

}  // namespace

std::string config_path() {
    if (const char* p = std::getenv("CPUSIM_CONFIG"); p && *p) return p;
    if (const char* x = std::getenv("XDG_CONFIG_HOME"); x && *x) return std::string(x) + "/cpusim/device.toml";
    return home_dir() + "/.config/cpusim/device.toml";
}

std::string socket_path(const Config& cfg) {
    if (!cfg.socket.empty()) return cfg.socket;
    if (const char* x = std::getenv("XDG_RUNTIME_DIR"); x && *x) return std::string(x) + "/cpusim/driver.sock";
    return "/tmp/cpusim-" + std::to_string(getuid()) + "/driver.sock";
}

}  // namespace cpusim
