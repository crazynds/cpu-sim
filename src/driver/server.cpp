#include "driver/server.h"


#include <fcntl.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <vector>

#include "common/spin.h"
#include "common/units.h"

namespace cpusim {

namespace {

std::string dirname_of(const std::string& path) {
    auto slash = path.rfind('/');
    return slash == std::string::npos ? "." : path.substr(0, slash);
}

std::string process_path(int pid) {
    char buf[4096];
    std::string link = "/proc/" + std::to_string(pid) + "/exe";
    ssize_t n = readlink(link.c_str(), buf, sizeof(buf) - 1);
    if (n <= 0) return "?";
    return std::string(buf, static_cast<size_t>(n));
}

template <size_t N>
void copy_str(char (&dst)[N], const std::string& src) {
    std::snprintf(dst, N, "%s", src.c_str());
}

int64_t to_unix(std::chrono::system_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch()).count();
}

// Copies between the client's address space and the VRAM, in chunks so that a
// partial transfer (a bad host pointer half way) is reported precisely.
Error transfer(int pid, uint8_t* local, uint64_t remote, uint64_t n, bool to_device) {
    constexpr uint64_t kChunk = 1ull << 30;
    while (n > 0) {
        size_t len = static_cast<size_t>(std::min(n, kChunk));
        iovec l{local, len};
        iovec r{reinterpret_cast<void*>(remote), len};
        ssize_t done = to_device ? process_vm_readv(pid, &l, 1, &r, 1, 0) : process_vm_writev(pid, &l, 1, &r, 1, 0);
        if (done < 0) {
            if (errno == EINTR) continue;
            if (errno == EPERM)
                std::fprintf(stderr, "cpusimd: process_vm access to pid %d denied (kernel.yama.ptrace_scope > 1?)\n", pid);
            return errno == EFAULT ? Error::InvalidValue : Error::OperatingSystem;
        }
        if (done == 0) return Error::InvalidValue;
        local += done;
        remote += static_cast<uint64_t>(done);
        n -= static_cast<uint64_t>(done);
    }
    return Error::Success;
}

}  // namespace

Server::Server(DeviceCore& core, Monitor& monitor) : core_(core), monitor_(monitor) {
    stop_fd_ = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
}

Server::~Server() {
    if (listen_fd_ >= 0) {
        close(listen_fd_);
        unlink(socket_path_.c_str());
    }
    if (lock_fd_ >= 0) {
        unlink(lock_path_.c_str());
        close(lock_fd_);
    }
    if (stop_fd_ >= 0) close(stop_fd_);
}

int Server::running_driver_pid(const std::string& socket_path) {
    std::string lock_path = socket_path + ".lock";
    int fd = open(lock_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    int pid = 0;
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {  // held: a driver is running
        char buf[32] = {};
        if (read(fd, buf, sizeof(buf) - 1) > 0) pid = std::atoi(buf);
    }
    close(fd);
    return pid;
}

bool Server::listen(const std::string& socket_path, std::string* error) {
    socket_path_ = socket_path;
    lock_path_ = socket_path + ".lock";

    std::string dir = dirname_of(socket_path);
    if (mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) {
        *error = "cannot create " + dir + ": " + std::strerror(errno);
        return false;
    }

    // One driver per socket. The lock dies with the process, so a crashed driver
    // never blocks the next one; the stale socket file is then replaced.
    lock_fd_ = open(lock_path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (lock_fd_ < 0) {
        *error = "cannot open " + lock_path_ + ": " + std::strerror(errno);
        return false;
    }
    if (flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) {
        *error = "another driver is already running";
        if (int pid = running_driver_pid(socket_path)) *error += " (pid " + std::to_string(pid) + ")";
        close(lock_fd_);
        lock_fd_ = -1;
        return false;
    }
    std::string pid = std::to_string(getpid()) + "\n";
    if (ftruncate(lock_fd_, 0) != 0 || write(lock_fd_, pid.data(), pid.size()) < 0) {
        *error = "cannot write " + lock_path_;
        return false;
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (socket_path.size() >= sizeof(addr.sun_path)) {
        *error = "socket path too long: " + socket_path;
        return false;
    }
    std::strcpy(addr.sun_path, socket_path.c_str());
    unlink(socket_path.c_str());

    listen_fd_ = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (listen_fd_ < 0 || bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        chmod(socket_path.c_str(), 0600) != 0 || ::listen(listen_fd_, 64) != 0) {
        *error = "cannot listen on " + socket_path + ": " + std::strerror(errno);
        return false;
    }
    return true;
}

void Server::request_stop() {
    uint64_t one = 1;
    ssize_t r = write(stop_fd_, &one, sizeof(one));
    (void)r;
}

void Server::run() {
    pollfd fds[2] = {{listen_fd_, POLLIN, 0}, {stop_fd_, POLLIN, 0}};
    while (true) {
        int n = poll(fds, 2, 1000);
        if (n < 0 && errno != EINTR) break;
        if (fds[1].revents & POLLIN) break;
        reap_finished();
        core_.cancel_launches(true);
        if (n <= 0 || !(fds[0].revents & POLLIN)) continue;

        int fd = accept4(listen_fd_, nullptr, nullptr, SOCK_CLOEXEC);
        if (fd < 0) continue;
        std::lock_guard lock(clients_mutex_);
        Client& c = clients_.emplace_back();
        c.fd = fd;
        c.thread = std::thread([this, &c] { serve(&c); });
    }

    // Wake every client thread and wait for it to free its context; a kernel
    // that never ends must not keep the driver from stopping.
    std::list<Client> remaining;
    {
        std::lock_guard lock(clients_mutex_);
        for (auto& c : clients_) shutdown(c.fd, SHUT_RDWR);
        remaining.splice(remaining.end(), clients_);
    }
    for (auto& c : remaining) {
        while (!c.done) {
            core_.cancel_launches(false);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        c.thread.join();
    }
}

void Server::reap_finished() {
    std::list<Client> finished;
    {
        std::lock_guard lock(clients_mutex_);
        for (auto it = clients_.begin(); it != clients_.end();) {
            auto next = std::next(it);
            if (it->done) finished.splice(finished.end(), clients_, it);
            it = next;
        }
    }
    for (auto& c : finished) c.thread.join();
}

void Server::serve(Client* client) {
    int fd = client->fd;
    ucred cred{};
    socklen_t len = sizeof(cred);
    getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &len);

    uint32_t ctx = 0;
    bool hello = false;
    bool attached = false;  // an extra connection of a context opened by another one
    std::vector<uint8_t> buffer(sizeof(proto::Request) + proto::kMaxPayload);
    while (true) {
        iovec iov{buffer.data(), buffer.size()};
        alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))];
        msghdr msg{};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
        ssize_t n = recvmsg_spinning(fd, &msg, MSG_CMSG_CLOEXEC);
        if (n < 0 && errno == EINTR) continue;
        int passed_fd = -1;
        for (cmsghdr* cm = CMSG_FIRSTHDR(&msg); cm; cm = CMSG_NXTHDR(&msg, cm))
            if (cm->cmsg_level == SOL_SOCKET && cm->cmsg_type == SCM_RIGHTS) std::memcpy(&passed_fd, CMSG_DATA(cm), sizeof(int));
        if (n < static_cast<ssize_t>(sizeof(proto::Request))) {
            if (passed_fd >= 0) close(passed_fd);
            break;
        }
        proto::Request req;
        std::memcpy(&req, buffer.data(), sizeof(req));
        const uint8_t* payload = buffer.data() + sizeof(req);
        size_t payload_size = size_t(n) - sizeof(req);
        if (req.magic != proto::kMagic) break;

        proto::Reply reply{};
        Error e = Error::Success;
        if (!hello && req.op != proto::Op::Hello) break;

        switch (req.op) {
            case proto::Op::Hello: {
                proto::HelloReply h{};
                if (hello || req.a != proto::kVersion || cred.uid != getuid()) {
                    h.header.error = static_cast<int32_t>(Error::InsufficientDriver);
                    send(fd, &h, sizeof(h), MSG_NOSIGNAL);
                    goto out;
                }
                if (req.b & proto::kHelloAttach) {
                    // Only the process that owns the context may join it.
                    bool owner = false;
                    for (const auto& info : core_.contexts()) owner |= info.id == req.c && info.pid == cred.pid;
                    if (!owner) {
                        h.header.error = static_cast<int32_t>(Error::InvalidValue);
                        send(fd, &h, sizeof(h), MSG_NOSIGNAL);
                        goto out;
                    }
                    ctx = uint32_t(req.c);
                    attached = true;
                } else if (!(req.b & proto::kHelloMonitor)) {
                    ctx = core_.open_context(cred.pid, process_path(cred.pid));
                    if (passed_fd >= 0) {  // the program's standard output, for device printf
                        core_.set_output(ctx, passed_fd);
                        passed_fd = -1;
                    }
                }
                hello = true;
                h.driver_pid = getpid();
                h.context = ctx;
                h.vram_base = reinterpret_cast<uint64_t>(core_.vram().base());
                h.vram_size = core_.vram().size();
                copy_str(h.driver_version, CPUSIM_VERSION);
                h.props = core_.properties();
                if (passed_fd >= 0) close(passed_fd);
                send(fd, &h, sizeof(h), MSG_NOSIGNAL);
                continue;
            }
            case proto::Op::Status: {
                auto st = std::make_unique<proto::DeviceStatus>();
                fill_status(st.get());
                send(fd, st.get(), sizeof(*st), MSG_NOSIGNAL);
                continue;
            }
            case proto::Op::Malloc:
                e = core_.allocate(ctx, req.a, &reply.a);
                break;
            case proto::Op::Free:
                e = core_.release(ctx, req.a);
                break;
            case proto::Op::MallocManaged:
                e = core_.allocate_managed(ctx, req.a, &reply.a);
                break;
            case proto::Op::SharedPool: {
                // The reply carries the pool's memory file for the client to map.
                if (core_.shared_fd() < 0) {
                    e = Error::NotSupported;
                    break;
                }
                reply.a = reinterpret_cast<uint64_t>(core_.shared_base());
                reply.b = core_.shared_size();
                int pool_fd = core_.shared_fd();
                iovec iov{&reply, sizeof(reply)};
                alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))];
                msghdr m{};
                m.msg_iov = &iov;
                m.msg_iovlen = 1;
                m.msg_control = control;
                m.msg_controllen = sizeof(control);
                cmsghdr* cm = CMSG_FIRSTHDR(&m);
                cm->cmsg_level = SOL_SOCKET;
                cm->cmsg_type = SCM_RIGHTS;
                cm->cmsg_len = CMSG_LEN(sizeof(int));
                std::memcpy(CMSG_DATA(cm), &pool_fd, sizeof(int));
                if (passed_fd >= 0) close(passed_fd);
                if (sendmsg(fd, &m, MSG_NOSIGNAL) < 0) goto out;
                continue;
            }
            case proto::Op::MemcpyHtoD:
            case proto::Op::MemcpyDtoH: {
                bool to_device = req.op == proto::Op::MemcpyHtoD;
                uint64_t dptr = to_device ? req.a : req.b;
                uint64_t hptr = to_device ? req.b : req.a;
                if (req.c == 0) break;
                uint8_t* local = nullptr;
                e = core_.translate(ctx, dptr, req.c, &local);
                if (e != Error::Success) break;
                BusyScope busy(core_.counters().copy_busy_ns);
                e = transfer(cred.pid, local, hptr, req.c, to_device);
                if (e == Error::Success)
                    (to_device ? core_.counters().bytes_htod : core_.counters().bytes_dtoh).fetch_add(req.c);
                break;
            }
            case proto::Op::MemcpyDtoD:
                e = core_.memcpy_dtod(ctx, req.a, req.b, req.c);
                break;
            case proto::Op::Memset:
                e = core_.memset(ctx, req.a, static_cast<int>(req.b), req.c);
                break;
            case proto::Op::MemGetInfo:
                core_.mem_info(&reply.a, &reply.b);
                break;
            case proto::Op::Synchronize:
                break;  // every operation is synchronous for now
            case proto::Op::LoadModule:
                if (passed_fd < 0) {
                    e = Error::InvalidValue;
                    break;
                }
                e = core_.load_module(ctx, passed_fd, &reply.a);
                break;
            case proto::Op::GetFunction:
                e = core_.get_function(ctx, req.a, std::string(reinterpret_cast<const char*>(payload), payload_size),
                                       &reply.a);
                break;
            case proto::Op::BindSymbols:
                e = core_.bind_symbols(ctx, req.a, std::string(reinterpret_cast<const char*>(payload), payload_size),
                                       req.b != 0);
                break;
            case proto::Op::Launch: {
                proto::LaunchPayload lp;
                if (payload_size < sizeof(lp)) {
                    e = Error::InvalidValue;
                    break;
                }
                std::memcpy(&lp, payload, sizeof(lp));
                if (payload_size != sizeof(lp) + lp.param_bytes) {
                    e = Error::InvalidValue;
                    break;
                }
                LaunchConfig cfg;
                std::memcpy(cfg.grid, lp.grid, sizeof(cfg.grid));
                std::memcpy(cfg.block, lp.block, sizeof(cfg.block));
                cfg.shared_bytes = lp.shared_bytes;
                cfg.profile = lp.profile != 0;
                cfg.debug = int(lp.debug);
                LaunchResult result;
                proto::LaunchReply lr{};
                e = core_.launch(ctx, req.a, cfg, payload + sizeof(lp), lp.param_bytes, &result);
                lr.header.error = static_cast<int32_t>(e);
                lr.kernel_ns = result.kernel_ns;
                std::memcpy(lr.counters, result.counters, sizeof(lr.counters));
                std::memcpy(lr.report, result.report, sizeof(lr.report));
                if (send(fd, &lr, sizeof(lr), MSG_NOSIGNAL) < 0) goto out;
                continue;
            }
            default:
                e = Error::NotSupported;
                break;
        }
        if (passed_fd >= 0) close(passed_fd);
        if (ctx == 0 && req.op != proto::Op::MemGetInfo) e = Error::InvalidValue;  // monitors cannot allocate
        reply.error = static_cast<int32_t>(e);
        if (send(fd, &reply, sizeof(reply), MSG_NOSIGNAL) < 0) break;
    }
out:
    if (ctx && !attached) core_.close_context(ctx);
    close(fd);
    std::lock_guard lock(clients_mutex_);
    client->done = true;
}

void Server::fill_status(proto::DeviceStatus* st) {
    *st = {};
    MonitorSample sample = monitor_.latest();
    const auto& sens = sample.sensors;
    const Config& cfg = core_.config();

    copy_str(st->driver_version, CPUSIM_VERSION);
    st->cuda_version = proto::kCudaVersion;
    st->driver_pid = getpid();
    st->driver_started_unix = to_unix(core_.started());
    copy_str(st->config_source, cfg.source);

    st->props = core_.properties();
    copy_str(st->cpu_model, core_.topology().cpu_model);
    copy_str(st->reserved_cpus, format_cpu_list(core_.reserved_cpus()));
    st->reserved_cores = static_cast<int32_t>(core_.reserved_cores().size());
    st->host_cores = static_cast<int32_t>(core_.topology().cores.size());
    st->host_logical_cpus = core_.topology().logical_cpus;

    core_.mem_info(&st->vram_free, &st->vram_total, &st->vram_largest_free);
    st->vram_used = st->vram_total - st->vram_free;
    st->vram_resident = sample.vram_resident;
    st->vram_shared_total = core_.shared_size();
    st->vram_shared_used = core_.shared_used();
    st->vram_base = reinterpret_cast<uint64_t>(core_.vram().base());
    st->page_size = core_.vram().page_size();
    copy_str(st->hugepages, hugepages_name(core_.vram().hugepages()));
    st->prefault = core_.vram().prefaulted();

    st->gpu_util = sample.gpu_util;
    st->memory_util = sample.memory_util;
    st->host_load = sens.host_load;
    st->temperature_c = sens.temperature_c;
    st->power_w = sens.power_w;
    st->clock_mhz = sens.clock_mhz;
    st->max_clock_mhz = sens.max_clock_mhz;
    st->fan_count = static_cast<int32_t>(std::min<size_t>(sens.fan_rpm.size(), proto::kMaxFans));
    for (int i = 0; i < st->fan_count; ++i) st->fan_rpm[i] = sens.fan_rpm[static_cast<size_t>(i)];
    st->pstate = sample.gpu_util > 0 ? 0 : sample.memory_util > 0 ? 2 : 8;

    const auto& c = core_.counters();
    st->kernels_launched = c.kernels_launched.load();
    st->kernels_profiled = c.kernels_profiled.load();
    st->bytes_htod = c.bytes_htod.load();
    st->bytes_dtoh = c.bytes_dtoh.load();
    st->bytes_dtod = c.bytes_dtod.load();
    st->bytes_memset = c.bytes_memset.load();
    st->alloc_calls = c.alloc_calls.load();
    st->free_calls = c.free_calls.load();
    st->failed_allocs = c.failed_allocs.load();

    for (const auto& ctx : core_.contexts()) {
        if (st->process_count == proto::kMaxProcesses) break;
        auto& p = st->processes[st->process_count++];
        p.pid = ctx.pid;
        p.context = ctx.id;
        p.used_bytes = ctx.used_bytes;
        p.allocations = ctx.allocations;
        p.started_unix = to_unix(ctx.started);
        copy_str(p.name, ctx.process_name);
    }
    // Applications on the in-process backend: their own copy of the device,
    // counted here so cpu-smi shows everything that runs on these cores.
    for (const auto& app : sample.inprocess) {
        st->vram_used = std::min(st->vram_total, st->vram_used + app.used_bytes);
        st->kernels_launched += app.kernels_launched;
        st->kernels_profiled += app.kernels_profiled;
        st->bytes_htod += app.bytes_htod;
        st->bytes_dtoh += app.bytes_dtoh;
        st->bytes_dtod += app.bytes_dtod;
        st->bytes_memset += app.bytes_memset;
        st->alloc_calls += app.alloc_calls;
        st->free_calls += app.free_calls;
        st->failed_allocs += app.failed_allocs;
        if (st->process_count == proto::kMaxProcesses) continue;
        auto& p = st->processes[st->process_count++];
        p.pid = app.pid;
        p.inprocess = 1;
        p.used_bytes = app.used_bytes;
        p.allocations = app.allocations;
        p.started_unix = app.started_unix;
        copy_str(p.name, app.name);
    }
    st->vram_free = st->vram_total - st->vram_used;
}

}  // namespace cpusim
