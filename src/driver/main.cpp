// cpusimd: the resident driver. Reserves cores and VRAM for the simulated GPU
// and serves applications (and cpu-smi) over a Unix socket.
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "common/config.h"
#include "common/paths.h"
#include "common/topology.h"
#include "common/units.h"
#include "device/device_core.h"
#include "driver/monitor.h"
#include "driver/server.h"

using namespace cpusim;

namespace {

Server* g_server = nullptr;

void on_signal(int) {
    if (g_server) g_server->request_stop();
}

void usage() {
    std::printf(
        "usage: cpusimd [options]\n"
        "\n"
        "Runs the CPU SIMT driver in the foreground until SIGINT/SIGTERM.\n"
        "\n"
        "  -c, --config PATH   device config (default: %s)\n"
        "  -s, --socket PATH   control socket (overrides the config)\n"
        "  -d, --daemon        detach and log to <socket>.log\n"
        "      --stop          stop the running driver\n"
        "      --status        print the pid of the running driver, exit 1 if none\n"
        "      --init-config   write the config file with this machine's defaults, commented out\n"
        "  -V, --version\n"
        "  -h, --help\n",
        config_path().c_str());
}

// Detaches before any thread exists. The parent waits until the child reports
// that it is listening (or failed), so `cpusimd -d` returns a meaningful status.
int g_ready_fd = -1;

bool daemonize(const std::string& log_path) {
    int pipefd[2];
    if (pipe(pipefd) != 0) return false;
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid > 0) {
        close(pipefd[1]);
        char ok = 0;
        _exit(read(pipefd[0], &ok, 1) == 1 && ok == 1 ? 0 : 1);
    }
    close(pipefd[0]);
    setsid();
    if (fork() > 0) _exit(0);
    g_ready_fd = pipefd[1];
    int null = open("/dev/null", O_RDONLY);
    int log = open(log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (null >= 0) dup2(null, 0);
    if (log >= 0) {
        dup2(log, 1);
        dup2(log, 2);
    }
    setvbuf(stdout, nullptr, _IOLBF, 0);
    return true;
}

// Writes the config file (never over an existing one) with every key commented
// out at the value the driver uses when the key is absent, so the file starts
// out equivalent to having none and each setting is one uncomment away.
int init_config() {
    const std::string path = config_path();
    if (access(path.c_str(), F_OK) == 0) {
        std::fprintf(stderr, "cpusimd: %s already exists; not overwriting it\n", path.c_str());
        return 1;
    }
    for (size_t at = path.find('/', 1); at != std::string::npos; at = path.find('/', at + 1))
        mkdir(path.substr(0, at).c_str(), 0755);

    const Topology topo = read_topology();
    const int cores = std::max(1, int(topo.cores.size()) - 2);
    std::vector<int> cpus;
    for (size_t i = topo.cores.size() - size_t(cores); i < topo.cores.size(); ++i)
        cpus.insert(cpus.end(), topo.cores[i].cpus.begin(), topo.cores[i].cpus.end());
    std::sort(cpus.begin(), cpus.end());
    const uint64_t ram = uint64_t(sysconf(_SC_PHYS_PAGES)) * uint64_t(sysconf(_SC_PAGESIZE));
    const uint64_t vram_gib = std::clamp<uint64_t>(ram / 4 / GiB, 1, 8);
    const Config defaults;

    std::ofstream f(path);
    f << "# CudaSimulator device configuration. Every key is optional: the values below,\n"
         "# commented out, are what the driver uses on this machine when the key is absent.\n"
         "# Restart the driver after editing (cpusimd --stop && cpusimd --daemon).\n"
         "\n"
         "[device]\n"
         "# name = \"" << short_cpu_model(topo.cpu_model) << "\"      # shown by cpu-smi and cudaGetDeviceProperties\n"
         "\n"
         "# Which CPUs run kernels: either a number of physical cores (\"SMs\", taken from\n"
         "# the end, leaving the first ones to the system) or a list of logical CPUs\n"
         "# (each listed CPU brings its whole physical core, SMT siblings included).\n"
         "# This machine: " << topo.cores.size() << " physical cores, " << topo.logical_cpus << " logical CPUs.\n"
         "# cores = " << cores << "\n"
         "# cpus = \"" << format_cpu_list(cpus) << "\"      # the same cores as above\n"
         "\n"
         "# Device memory (cudaMalloc): a quarter of the RAM, at most 8 GiB.\n"
         "# vram_private = \"" << vram_gib << "GiB\"\n"
         "# Shared pool for cudaMallocManaged and mapped host memory (\"0\" disables it).\n"
         "# vram_shared = \"" << defaults.vram_shared / GiB << "GiB\"\n"
         "# hugepages = \"" << hugepages_name(defaults.hugepages) << "\"      # 2MiB, 1GiB or none\n"
         "# prefault = " << (defaults.prefault ? "true" : "false") << "      # true: occupy all of the VRAM in RAM at startup\n"
         "\n"
         "[driver]\n"
         "# socket = \"" << socket_path(defaults) << "\"\n";
    if (!f) {
        std::fprintf(stderr, "cpusimd: cannot write %s\n", path.c_str());
        return 1;
    }
    std::printf("wrote %s\n", path.c_str());
    return 0;
}

void report_ready(bool ok) {
    if (g_ready_fd < 0) return;
    char c = ok ? 1 : 0;
    ssize_t r = write(g_ready_fd, &c, 1);
    (void)r;
    close(g_ready_fd);
    g_ready_fd = -1;
}

}  // namespace

int main(int argc, char** argv) {
    bool detach = false, stop = false, status = false;
    std::string socket_override;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto value = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "cpusimd: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-c" || a == "--config") setenv("CPUSIM_CONFIG", value(), 1);
        else if (a == "-s" || a == "--socket") socket_override = value();
        else if (a == "-d" || a == "--daemon") detach = true;
        else if (a == "--stop") stop = true;
        else if (a == "--status") status = true;
        else if (a == "--init-config") return init_config();
        else if (a == "-V" || a == "--version") return std::printf("cpusimd %s\n", CPUSIM_VERSION), 0;
        else if (a == "-h" || a == "--help") return usage(), 0;
        else {
            std::fprintf(stderr, "cpusimd: unknown option '%s'\n", a.c_str());
            usage();
            return 2;
        }
    }

    std::string err;
    auto cfg = load_config(&err);
    if (!cfg) {
        std::fprintf(stderr, "cpusimd: %s\n", err.c_str());
        return 1;
    }
    if (!socket_override.empty()) cfg->socket = socket_override;
    std::string sock = socket_path(*cfg);

    if (stop || status) {
        int pid = Server::running_driver_pid(sock);
        if (!pid) {
            std::fprintf(stderr, "cpusimd: no driver running on %s\n", sock.c_str());
            return 1;
        }
        if (status) return std::printf("%d\n", pid), 0;
        if (kill(pid, SIGTERM) != 0) {
            std::fprintf(stderr, "cpusimd: kill %d: %s\n", pid, std::strerror(errno));
            return 1;
        }
        for (int i = 0; i < 100 && Server::running_driver_pid(sock) == pid; ++i) usleep(50000);
        return 0;
    }

    auto core = DeviceCore::create(*cfg, kPreferredVramBase, &err);
    if (!core) {
        std::fprintf(stderr, "cpusimd: cannot create the device: %s\n", err.c_str());
        return 1;
    }

    if (detach) {
        if (Server::running_driver_pid(sock)) {
            std::fprintf(stderr, "cpusimd: another driver is already running on %s\n", sock.c_str());
            return 1;
        }
        std::printf("cpusimd: detaching, log in %s.log\n", sock.c_str());
        std::fflush(stdout);
        if (!daemonize(sock + ".log")) {
            std::fprintf(stderr, "cpusimd: fork failed\n");
            return 1;
        }
    }

    Monitor monitor(*core);
    Server server(*core, monitor);
    if (!server.listen(sock, &err)) {
        std::fprintf(stderr, "cpusimd: %s\n", err.c_str());
        report_ready(false);
        return 1;
    }

    const auto& p = core->properties();
    std::printf("cpusimd %s: device 0 \"%s\", compute capability %d.%d\n", CPUSIM_VERSION, p.name, p.major, p.minor);
    std::printf("  SMs (physical cores): %d, logical CPUs %s\n", p.multiProcessorCount,
                format_cpu_list(core->reserved_cpus()).c_str());
    std::printf("  VRAM: %s at %p, pages %s%s\n", format_size(core->vram().size()).c_str(),
                static_cast<void*>(core->vram().base()), hugepages_name(core->vram().hugepages()),
                core->vram().prefaulted() ? ", prefaulted" : "");
    std::printf("  config: %s\n  socket: %s\n", cfg->source.c_str(), sock.c_str());
    std::fflush(stdout);
    report_ready(true);

    g_server = &server;
    struct sigaction sa{};
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    signal(SIGPIPE, SIG_IGN);

    server.run();
    g_server = nullptr;
    std::printf("cpusimd: stopped\n");
    return 0;
}
