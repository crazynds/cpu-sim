#include "device/debug_exec.h"

#include <dlfcn.h>
#include <setjmp.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace cpusim {

namespace {

const size_t kPage = size_t(sysconf(_SC_PAGESIZE));

// `size` usable bytes whose END touches a guard page (overflows fault on the
// first byte past it) and with a guard page before the start.
struct Guarded {
    uint8_t* map = nullptr;
    size_t map_size = 0;
    uint8_t* data = nullptr;
    size_t size = 0;

    explicit Guarded(size_t bytes, size_t align = 64) {
        size = (bytes + align - 1) / align * align;
        size_t body = (size + kPage - 1) / kPage * kPage;
        map_size = body + 2 * kPage;
        map = static_cast<uint8_t*>(mmap(nullptr, map_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        mprotect(map + kPage, body, PROT_READ | PROT_WRITE);
        data = map + kPage + body - size;
    }
    ~Guarded() {
        if (map) munmap(map, map_size);
    }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    // Any address within one page of the buffer counts as "near" it.
    bool near(uintptr_t a) const {
        return a + kPage >= uintptr_t(data) && a < uintptr_t(data) + size + kPage;
    }
};

// What the current step is doing, for the fault handler.
struct StepInfo {
    bool active = false;
    sigjmp_buf jump;
    uint64_t step = 0;
    uint32_t block[3] = {};
    uint32_t warp = 0;
    uint32_t state = 0;
    uintptr_t fault_addr = 0;
    uintptr_t fault_ip = 0;
    int signal = 0;
};

thread_local StepInfo* t_step = nullptr;
std::once_flag g_handlers_once;
struct sigaction g_old_segv, g_old_bus, g_old_fpe;

void on_fault(int sig, siginfo_t* info, void* uctx) {
    StepInfo* s = t_step;
    if (!s || !s->active) {
        // Not ours: behave as if we were never installed.
        struct sigaction* old = sig == SIGSEGV ? &g_old_segv : sig == SIGBUS ? &g_old_bus : &g_old_fpe;
        sigaction(sig, old, nullptr);
        raise(sig);
        return;
    }
    s->signal = sig;
    s->fault_addr = uintptr_t(info->si_addr);
    s->fault_ip = uintptr_t(static_cast<ucontext_t*>(uctx)->uc_mcontext.gregs[REG_RIP]);
    s->active = false;
    siglongjmp(s->jump, 1);
}

void install_handlers() {
    std::call_once(g_handlers_once, [] {
        // SA_ONSTACK: the kernel may have wrecked its stack; run_debug sets the
        // alternate stack of the thread that runs it.
        struct sigaction sa{};
        sa.sa_sigaction = on_fault;
        sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
        sigaction(SIGSEGV, &sa, &g_old_segv);
        sigaction(SIGBUS, &sa, &g_old_bus);
        sigaction(SIGFPE, &sa, &g_old_fpe);
    });
}

std::string where_in_code(uintptr_t ip) {
    Dl_info info{};
    char buf[512];
    if (dladdr(reinterpret_cast<void*>(ip), &info) && info.dli_fname) {
        std::snprintf(buf, sizeof(buf), "%s+0x%lx (in %s+0x%lx)", info.dli_sname ? info.dli_sname : "?",
                      (unsigned long)(ip - uintptr_t(info.dli_saddr)), info.dli_fname,
                      (unsigned long)(ip - uintptr_t(info.dli_fbase)));
    } else {
        std::snprintf(buf, sizeof(buf), "0x%lx", (unsigned long)ip);
    }
    return buf;
}

// One step under the fault handler. Kept apart so that no local of the caller
// lives across sigsetjmp. Returns false if the step faulted.
__attribute__((noinline)) bool run_step(cpusim_entry_fn entry, cpusim_warp_ctx* c, StepInfo* step, uint32_t* reason) {
    if (sigsetjmp(step->jump, 1) != 0) return false;
    step->active = true;
    *reason = entry(c);
    step->active = false;
    return true;
}

}  // namespace

bool run_debug(cpusim_entry_fn entry, const cpusim_kernel_meta& meta, const LaunchConfig& cfg, const uint8_t* params,
               uint64_t* counters, bool trace, std::string* report) {
    install_handlers();
    // An alternate signal stack is per thread.
    std::vector<uint8_t> alt(1 << 16);
    stack_t ss{};
    ss.ss_sp = alt.data();
    ss.ss_size = alt.size();
    stack_t old_ss{};
    sigaltstack(&ss, &old_ss);

    const uint32_t threads = cfg.block[0] * cfg.block[1] * cfg.block[2];
    const uint32_t nwarps = (threads + CPUSIM_WARP_SIZE - 1) / CPUSIM_WARP_SIZE;
    const uint32_t frame_bytes = (meta.frame_bytes + 63) / 64 * 64;

    Guarded shared(std::max<uint32_t>(cfg.shared_bytes, 1), 16);
    std::vector<std::unique_ptr<Guarded>> frames;
    for (uint32_t w = 0; w < nwarps && frame_bytes; ++w) frames.push_back(std::make_unique<Guarded>(frame_bytes));

    CtxArray ctxs = Executor::make_groups(cfg, CPUSIM_WARP_SIZE);
    for (uint32_t w = 0; w < nwarps; ++w) {
        cpusim_warp_ctx& c = *ctxs.at(w);
        c.params = params;
        c.shared = shared.data;
        c.counters = counters;
        c.frame = frame_bytes ? frames[w]->data : nullptr;
    }

    StepInfo step;
    t_step = &step;
    bool ok = true;
    const uint64_t blocks = uint64_t(cfg.grid[0]) * cfg.grid[1] * cfg.grid[2];
    std::vector<uint8_t> done(nwarps);

    for (uint64_t b = 0; b < blocks && ok; ++b) {
        uint32_t idx[3] = {uint32_t(b % cfg.grid[0]), uint32_t(b / cfg.grid[0] % cfg.grid[1]),
                           uint32_t(b / cfg.grid[0] / cfg.grid[1])};
        for (uint32_t w = 0; w < nwarps; ++w) {
            cpusim_warp_ctx& c = *ctxs.at(w);
            std::memcpy(c.block_idx, idx, sizeof(idx));
            if (c.frame) *reinterpret_cast<uint32_t*>(c.frame) = 0;
        }
        std::fill(done.begin(), done.end(), 0);
        uint32_t remaining = nwarps;
        while (remaining && ok) {
            for (uint32_t w = 0; w < nwarps && ok; ++w) {
                if (done[w]) continue;
                cpusim_warp_ctx& c = *ctxs.at(w);
                step.step += 1;
                std::memcpy(step.block, idx, sizeof(idx));
                step.warp = w;
                step.state = c.frame ? *reinterpret_cast<uint32_t*>(c.frame) : 0;
                uint32_t r = 0;
                if (run_step(entry, &c, &step, &r)) {
                    if (trace)
                        std::fprintf(stderr, "cpusim debug: step %llu block (%u,%u,%u) warp %u from state %u -> %s\n",
                                     (unsigned long long)step.step, idx[0], idx[1], idx[2], w, step.state,
                                     r == CPUSIM_EXIT ? "exit" : "barrier");
                    if (r == CPUSIM_EXIT) {
                        done[w] = 1;
                        --remaining;
                    } else if (!c.frame) {
                        ok = false;
                        *report = "kernel returned BARRIER but has no frame";
                    }
                } else {
                    ok = false;
                }
            }
        }
    }
    t_step = nullptr;
    sigaltstack(&old_ss, nullptr);

    if (!ok && report->empty()) {
        char buf[1024];
        std::string region = "outside shared memory and warp frames";
        uintptr_t a = step.fault_addr;
        if (shared.near(a)) {
            long off = long(a) - long(uintptr_t(shared.data));
            std::snprintf(buf, sizeof(buf), "shared memory offset %ld (size %zu)", off, size_t(cfg.shared_bytes));
            region = buf;
        }
        for (uint32_t w = 0; w < frames.size(); ++w) {
            if (frames[w]->near(a)) {
                long off = long(a) - long(uintptr_t(frames[w]->data));
                std::snprintf(buf, sizeof(buf), "frame of warp %u offset %ld (frame size %u)", w, off, meta.frame_bytes);
                region = buf;
            }
        }
        std::snprintf(buf, sizeof(buf),
                      "%s at step %llu: block (%u,%u,%u) warp %u resuming from state %u\n"
                      "  address 0x%lx: %s\n  code    %s",
                      step.signal == SIGFPE ? "arithmetic fault" : "invalid memory access",
                      (unsigned long long)step.step, step.block[0], step.block[1], step.block[2], step.warp, step.state,
                      (unsigned long)a, region.c_str(), where_in_code(step.fault_ip).c_str());
        *report = buf;
    }
    return ok;
}

}  // namespace cpusim
