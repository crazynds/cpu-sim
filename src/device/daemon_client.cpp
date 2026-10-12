#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <vector>

#include "common/config.h"
#include "common/paths.h"
#include "common/spin.h"
#include "device/device.h"
#include "device/driver_client.h"


namespace cpusim {

DriverConnection::~DriverConnection() {
    if (fd_ >= 0) close(fd_);
}

Error DriverConnection::connect(const std::string& path, std::string* message) {
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        *message = "socket path too long: " + path;
        return Error::InitializationError;
    }
    std::strcpy(addr.sun_path, path.c_str());

    fd_ = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (fd_ < 0) {
        *message = std::string("socket: ") + std::strerror(errno);
        return Error::OperatingSystem;
    }
    if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        int err = errno;
        close(fd_);
        fd_ = -1;
        *message = "cannot reach the driver at " + path + ": " + std::strerror(err);
        return Error::NoDevice;
    }
    return Error::Success;
}

Error DriverConnection::call(const proto::Request& req, void* reply, size_t reply_size, std::string* message,
                              const void* payload, size_t payload_size, int fd, int* received_fd) {
    if (payload_size > proto::kMaxPayload) {
        *message = "request payload too large";
        return Error::InvalidValue;
    }
    iovec iov[2] = {{const_cast<proto::Request*>(&req), sizeof(req)}, {const_cast<void*>(payload), payload_size}};
    msghdr msg{};
    msg.msg_iov = iov;
    msg.msg_iovlen = payload_size ? 2 : 1;
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))];
    if (fd >= 0) {
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
        cmsghdr* cm = CMSG_FIRSTHDR(&msg);
        cm->cmsg_level = SOL_SOCKET;
        cm->cmsg_type = SCM_RIGHTS;
        cm->cmsg_len = CMSG_LEN(sizeof(int));
        std::memcpy(CMSG_DATA(cm), &fd, sizeof(int));
    }
    if (sendmsg(fd_, &msg, MSG_NOSIGNAL) != static_cast<ssize_t>(sizeof(req) + payload_size)) {
        *message = std::string("lost the connection to the driver: ") + std::strerror(errno);
        return Error::Unknown;
    }
    // The reply may carry a file descriptor (SharedPool).
    iovec riov{reply, reply_size};
    alignas(cmsghdr) char rcontrol[CMSG_SPACE(sizeof(int))];
    msghdr rmsg{};
    rmsg.msg_iov = &riov;
    rmsg.msg_iovlen = 1;
    rmsg.msg_control = rcontrol;
    rmsg.msg_controllen = sizeof(rcontrol);
    ssize_t n;
    do n = recvmsg_spinning(fd_, &rmsg, MSG_CMSG_CLOEXEC);
    while (n < 0 && errno == EINTR);
    for (cmsghdr* cm = CMSG_FIRSTHDR(&rmsg); n > 0 && cm; cm = CMSG_NXTHDR(&rmsg, cm)) {
        if (cm->cmsg_level != SOL_SOCKET || cm->cmsg_type != SCM_RIGHTS) continue;
        int got;
        std::memcpy(&got, CMSG_DATA(cm), sizeof(int));
        if (received_fd) *received_fd = got;
        else close(got);
    }
    if (n < static_cast<ssize_t>(sizeof(proto::Reply))) {
        *message = n == 0 ? "the driver closed the connection" : std::string("recv: ") + std::strerror(errno);
        return Error::Unknown;
    }
    return static_cast<Error>(static_cast<proto::Reply*>(reply)->error);
}

namespace {

std::string resolve_socket(std::string* message, bool* ok) {
    std::string err;
    auto cfg = load_config(&err);
    *ok = cfg.has_value();
    if (!cfg) {
        *message = err;
        return {};
    }
    return socket_path(*cfg);
}

// The VRAM lives in the driver. Copies go straight between the application's
// memory and the VRAM through process_vm_readv/writev, done by the driver
// (one copy, the role of the DMA engine).
class DaemonDevice final : public Device {
public:
    DaemonDevice() = default;
    ~DaemonDevice() override {
        if (reserved_) munmap(reinterpret_cast<void*>(vram_base_), vram_size_);
        if (shared_base_) munmap(reinterpret_cast<void*>(shared_base_), shared_size_);
    }

    Error init(const std::string& path, std::string* message, bool quiet_if_absent) {
        Error e = conn_.connect(path, message);
        if (e != Error::Success) {
            if (e == Error::NoDevice && !quiet_if_absent) *message += " (is cpusimd running?)";
            return e;
        }
        proto::HelloReply hello{};
        // Our standard output goes along: the kernels' printf writes there.
        e = conn_.call({proto::kMagic, proto::Op::Hello, proto::kVersion, 0, 0}, &hello, sizeof(hello), message, nullptr,
                       0, STDOUT_FILENO);
        if (e != Error::Success) {
            if (message->empty()) *message = "the driver rejected this client (version mismatch?)";
            return e;
        }
        props_ = hello.props;
        driver_version_ = hello.driver_version;
        path_ = path;
        context_ = hello.context;
        vram_base_ = hello.vram_base;
        vram_size_ = hello.vram_size;

        // Ubuntu's Yama (ptrace_scope = 1) only lets an ancestor read our memory;
        // allow the driver explicitly. Scope 2 or 3 needs the staging fallback (TODO).
        if (prctl(PR_SET_PTRACER, static_cast<unsigned long>(hello.driver_pid), 0, 0, 0) != 0 && errno != EINVAL) {
            *message = std::string("prctl(PR_SET_PTRACER): ") + std::strerror(errno);
            return Error::InitializationError;
        }

        // The shared pool (cudaMallocManaged): the driver's memory file, mapped
        // at the driver's address. If that range is taken here, managed memory
        // is not available to this process.
        {
            proto::Reply r{};
            int pool_fd = -1;
            std::string msg;
            if (conn_.call({proto::kMagic, proto::Op::SharedPool, 0, 0, 0}, &r, sizeof(r), &msg, nullptr, 0, -1,
                           &pool_fd) == Error::Success &&
                pool_fd >= 0) {
                void* p = mmap(reinterpret_cast<void*>(r.a), r.b, PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_FIXED_NOREPLACE, pool_fd, 0);
                if (p == reinterpret_cast<void*>(r.a)) {
                    shared_base_ = r.a;
                    shared_size_ = r.b;
                } else if (p != MAP_FAILED) {
                    munmap(p, r.b);
                }
            }
            if (pool_fd >= 0) close(pool_fd);
        }

        // Keep our own allocations out of the driver's VRAM range, so a device
        // pointer is never also a valid host pointer here (unified addressing).
        void* p = mmap(reinterpret_cast<void*>(vram_base_), vram_size_, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
        if (p == reinterpret_cast<void*>(vram_base_)) {
            reserved_ = true;
        } else if (p != MAP_FAILED) {
            munmap(p, vram_size_);
        }
        return Error::Success;
    }

    const char* backend_name() const override { return "daemon"; }
    const DeviceProperties& properties() const override { return props_; }
    std::string driver_version() const override { return driver_version_; }

    Error malloc(uint64_t bytes, uint64_t* dptr) override {
        proto::Reply r{};
        Error e = call(proto::Op::Malloc, bytes, 0, 0, &r);
        *dptr = e == Error::Success ? r.a : 0;
        return e;
    }
    Error free(uint64_t dptr) override { return call(proto::Op::Free, dptr, 0, 0); }
    Error malloc_managed(uint64_t bytes, uint64_t* dptr) override {
        *dptr = 0;
        if (!shared_base_) return Error::NotSupported;
        proto::Reply r{};
        Error e = call(proto::Op::MallocManaged, bytes, 0, 0, &r);
        if (e == Error::Success) *dptr = r.a;
        return e;
    }
    bool is_managed_pointer(const void* ptr) const override {
        auto p = reinterpret_cast<uint64_t>(ptr);
        return shared_base_ && p >= shared_base_ && p < shared_base_ + shared_size_;
    }

    Error memcpy_htod(uint64_t dst, const void* src, uint64_t n) override {
        return call(proto::Op::MemcpyHtoD, dst, reinterpret_cast<uint64_t>(src), n);
    }
    Error memcpy_dtoh(void* dst, uint64_t src, uint64_t n) override {
        return call(proto::Op::MemcpyDtoH, reinterpret_cast<uint64_t>(dst), src, n);
    }
    Error memcpy_dtod(uint64_t dst, uint64_t src, uint64_t n) override { return call(proto::Op::MemcpyDtoD, dst, src, n); }
    Error memset(uint64_t dst, int value, uint64_t n) override {
        return call(proto::Op::Memset, dst, static_cast<uint64_t>(static_cast<uint8_t>(value)), n);
    }

    Error mem_get_info(uint64_t* free, uint64_t* total) override {
        proto::Reply r{};
        Error e = call(proto::Op::MemGetInfo, 0, 0, 0, &r);
        if (e == Error::Success) {
            *free = r.a;
            *total = r.b;
        }
        return e;
    }

    Error synchronize() override { return call(proto::Op::Synchronize, 0, 0, 0); }

    Error load_module(const void* image, size_t size, uint64_t* module) override {
        int fd = make_memfd("cpusim-module", image, size);
        if (fd < 0) return Error::OperatingSystem;
        proto::Reply r{};
        Error e = call(proto::Op::LoadModule, 0, 0, 0, &r, nullptr, 0, fd);
        close(fd);
        *module = r.a;
        return e;
    }

    Error get_function(uint64_t module, const std::string& name, uint64_t* function) override {
        proto::Reply r{};
        Error e = call(proto::Op::GetFunction, module, 0, 0, &r, name.data(), name.size());
        *function = r.a;
        return e;
    }

    Error bind_symbols(uint64_t module, const std::string& table, bool init) override {
        if (table.size() > proto::kMaxPayload) return Error::NotSupported;
        return call(proto::Op::BindSymbols, module, init ? 1 : 0, 0, nullptr, table.data(), table.size());
    }

    Error launch(uint64_t function, const LaunchConfig& cfg, const void* params, size_t param_bytes,
                 LaunchResult* result) override {
        std::vector<uint8_t> payload(sizeof(proto::LaunchPayload) + param_bytes);
        proto::LaunchPayload lp{};
        std::memcpy(lp.grid, cfg.grid, sizeof(lp.grid));
        std::memcpy(lp.block, cfg.block, sizeof(lp.block));
        lp.shared_bytes = cfg.shared_bytes;
        lp.param_bytes = uint32_t(param_bytes);
        lp.profile = cfg.profile;
        lp.debug = uint32_t(cfg.debug);
        std::memcpy(payload.data(), &lp, sizeof(lp));
        if (param_bytes) std::memcpy(payload.data() + sizeof(lp), params, param_bytes);
        proto::LaunchReply r{};
        std::string message;
        Lease lease(*this);
        Error e = lease.conn->call({proto::kMagic, proto::Op::Launch, function, 0, 0}, &r, sizeof(r), &message,
                                   payload.data(), payload.size());
        if (e == Error::Unknown && !message.empty()) std::fprintf(stderr, "cpusim: %s\n", message.c_str());
        if (result) {
            result->kernel_ns = r.kernel_ns;
            std::memcpy(result->counters, r.counters, sizeof(r.counters));
            std::memcpy(result->report, r.report, sizeof(r.report));
        }
        return e;
    }

    bool is_device_pointer(const void* ptr) const override {
        auto p = reinterpret_cast<uint64_t>(ptr);
        return reserved_ && p >= vram_base_ && p < vram_base_ + vram_size_;
    }

private:
    Error call(proto::Op op, uint64_t a, uint64_t b, uint64_t c, proto::Reply* out = nullptr,
               const void* payload = nullptr, size_t payload_size = 0, int fd = -1) {
        proto::Reply r{};
        std::string message;
        Lease lease(*this);  // one request in flight per connection
        Error e = lease.conn->call({proto::kMagic, op, a, b, c}, &r, sizeof(r), &message, payload, payload_size, fd);
        if (e == Error::Unknown && !message.empty()) std::fprintf(stderr, "cpusim: %s\n", message.c_str());
        if (out) *out = r;
        return e;
    }

    // A connection for one request: an idle one, or a new one attached to
    // our context (several streams issue requests at the same time).
    struct Lease {
        DaemonDevice& dev;
        DriverConnection* conn = nullptr;
        explicit Lease(DaemonDevice& d) : dev(d) {
            {
                std::lock_guard lock(dev.mutex_);
                if (!dev.idle_.empty()) {
                    conn = dev.idle_.back();
                    dev.idle_.pop_back();
                    return;
                }
            }
            auto fresh = std::make_unique<DriverConnection>();
            std::string msg;
            proto::HelloReply h{};
            if (fresh->connect(dev.path_, &msg) == Error::Success &&
                fresh->call({proto::kMagic, proto::Op::Hello, proto::kVersion, proto::kHelloAttach, dev.context_}, &h,
                            sizeof(h), &msg) == Error::Success) {
                std::lock_guard lock(dev.mutex_);
                dev.extra_.push_back(std::move(fresh));
                conn = dev.extra_.back().get();
                return;
            }
            // Could not open another one: wait for the main connection.
            dev.main_mutex_.lock();
            conn = &dev.conn_;
        }
        ~Lease() {
            if (conn == &dev.conn_) {
                dev.main_mutex_.unlock();
                return;
            }
            std::lock_guard lock(dev.mutex_);
            dev.idle_.push_back(conn);
        }
    };

    DriverConnection conn_;  // the context's own connection (Hello, setup)
    std::mutex main_mutex_;   // conn_ in use by a request
    std::mutex mutex_;        // idle_ and extra_
    std::vector<std::unique_ptr<DriverConnection>> extra_;  // attached to our context
    std::vector<DriverConnection*> idle_;
    std::string path_;
    uint32_t context_ = 0;
    DeviceProperties props_{};
    std::string driver_version_;
    uint64_t vram_base_ = 0, vram_size_ = 0;
    bool reserved_ = false;
    uint64_t shared_base_ = 0, shared_size_ = 0;  // the shared pool, mapped here (0: not available)
};

}  // namespace

std::unique_ptr<Device> open_daemon_device(Error* error, std::string* message, bool quiet_if_absent) {
    bool ok = false;
    std::string path = resolve_socket(message, &ok);
    if (!ok) {
        *error = Error::InitializationError;
        return nullptr;
    }
    auto dev = std::make_unique<DaemonDevice>();
    *error = dev->init(path, message, quiet_if_absent);
    if (*error != Error::Success) return nullptr;
    return dev;
}

Error query_driver_status(proto::DeviceStatus* status, std::string* message) {
    bool ok = false;
    std::string path = resolve_socket(message, &ok);
    if (!ok) return Error::InitializationError;

    DriverConnection conn;
    if (Error e = conn.connect(path, message); e != Error::Success) return e;
    proto::HelloReply hello{};
    Error e = conn.call({proto::kMagic, proto::Op::Hello, proto::kVersion, proto::kHelloMonitor, 0}, &hello,
                        sizeof(hello), message);
    if (e != Error::Success) return e;
    return conn.call({proto::kMagic, proto::Op::Status, 0, 0, 0}, status, sizeof(*status), message);
}

}  // namespace cpusim
