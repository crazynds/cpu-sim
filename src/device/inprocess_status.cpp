#include "device/inprocess_status.h"

#include <time.h>

#include "common/paths.h"

namespace cpusim {

std::string inprocess_status_dir(const Config& cfg) {
    std::string socket = socket_path(cfg);
    size_t slash = socket.rfind('/');
    return (slash == std::string::npos ? std::string(".") : socket.substr(0, slash)) + "/inprocess";
}

uint64_t monotonic_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

uint64_t busy_now_ns(uint64_t completed, uint64_t since) {
    if (since == 0) return completed;
    uint64_t now = monotonic_ns();
    return completed + (now > since ? now - since : 0);
}

}  // namespace cpusim
