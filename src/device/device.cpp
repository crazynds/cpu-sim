#include "device/device.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cstdlib>
#include <string>

namespace cpusim {

int make_memfd(const char* name, const void* image, size_t size) {
    int fd = memfd_create(name, MFD_CLOEXEC);
    if (fd < 0) return -1;
    auto* p = static_cast<const char*>(image);
    while (size > 0) {
        ssize_t n = write(fd, p, size);
        if (n <= 0) {
            close(fd);
            return -1;
        }
        p += n;
        size -= size_t(n);
    }
    return fd;
}

std::unique_ptr<Device> open_device(Error* error, std::string* message) {
    const char* env = std::getenv("CPUSIM_BACKEND");
    std::string backend = env ? env : "auto";
    if (backend == "daemon") return open_daemon_device(error, message, false);
    if (backend == "inprocess") return open_inprocess_device(error, message);
    if (backend == "auto") {
        if (auto dev = open_daemon_device(error, message, true)) return dev;
        if (*error != Error::NoDevice) return nullptr;  // the driver is there but refused us
        return open_inprocess_device(error, message);
    }
    *error = Error::InitializationError;
    *message = "unknown CPUSIM_BACKEND '" + backend + "' (expected daemon, inprocess or auto)";
    return nullptr;
}

}  // namespace cpusim
