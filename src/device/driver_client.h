#pragma once

#include <cstddef>
#include <string>

#include "common/error.h"
#include "device/protocol.h"

namespace cpusim {

// A connection to the driver's control socket.
class DriverConnection {
public:
    DriverConnection() = default;
    ~DriverConnection();
    DriverConnection(const DriverConnection&) = delete;
    DriverConnection& operator=(const DriverConnection&) = delete;

    // Error::NoDevice when nothing is listening.
    Error connect(const std::string& socket_path, std::string* message);

    // Sends one request (with an optional payload and file descriptor) and waits
    // for its reply (at least sizeof(proto::Reply) bytes).
    Error call(const proto::Request& req, void* reply, size_t reply_size, std::string* message,
               const void* payload = nullptr, size_t payload_size = 0, int fd = -1, int* received_fd = nullptr);

    bool connected() const { return fd_ >= 0; }

private:
    int fd_ = -1;
};

// One-shot status query for monitoring tools.
Error query_driver_status(proto::DeviceStatus* status, std::string* message);

}  // namespace cpusim
