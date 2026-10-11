#pragma once

#include <list>
#include <mutex>
#include <string>
#include <thread>

#include "device/device_core.h"
#include "device/protocol.h"
#include "driver/monitor.h"

namespace cpusim {

// Accepts applications and monitoring tools on the control socket, one thread
// per connection. Each application gets a context; when its connection closes
// (exit or crash), everything it allocated is freed.
class Server {
public:
    Server(DeviceCore& core, Monitor& monitor);
    ~Server();

    // Creates the socket directory, takes the single-instance lock and listens.
    bool listen(const std::string& socket_path, std::string* error);

    // Serves until request_stop() (safe to call from a signal handler).
    void run();
    void request_stop();

    // Pid of the driver holding the lock at `socket_path`, or 0.
    static int running_driver_pid(const std::string& socket_path);

private:
    struct Client {
        int fd = -1;
        std::thread thread;
        bool done = false;
    };

    void serve(Client* client);
    void fill_status(proto::DeviceStatus* st);
    void reap_finished();

    DeviceCore& core_;
    Monitor& monitor_;
    std::string socket_path_, lock_path_;
    int listen_fd_ = -1, lock_fd_ = -1, stop_fd_ = -1;

    std::mutex clients_mutex_;
    std::list<Client> clients_;
};

}  // namespace cpusim
