#include "device/executor.h"

#include <pthread.h>
#include <sched.h>
#include <setjmp.h>
#include <signal.h>
#include <sys/mman.h>
#include <ucontext.h>
#include <unistd.h>

#include <immintrin.h>

#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>

#include "common/spin.h"
#include "common/topology.h"

namespace cpusim {

using AlignedAllocator = AlignedAlloc<uint8_t>;

namespace {

// Where a worker resumes when the kernel it runs faults.
struct Recovery {
    sigjmp_buf jump;
    KernelFault fault;
};
thread_local Recovery* t_recovery = nullptr;
// Set only while the worker runs kernel code: a fault there is the kernel's,
// and a cancellation may take the worker out of it.
thread_local volatile sig_atomic_t t_in_kernel = 0;
thread_local const std::atomic<bool>* t_aborted = nullptr;  // of the job the worker runs
struct sigaction g_previous[NSIG];
constexpr int kCancelSignal = SIGUSR2;

inline uint32_t in_kernel(cpusim_entry_fn entry, cpusim_warp_ctx* c) {
    t_in_kernel = 1;
    uint32_t r = entry(c);
    t_in_kernel = 0;
    return r;
}

// A fault outside a kernel run: what would have happened without us.
void forward(int sig, siginfo_t* info, void* uctx) {
    const struct sigaction& p = g_previous[sig];
    if ((p.sa_flags & SA_SIGINFO) && p.sa_sigaction) return p.sa_sigaction(sig, info, uctx);
    if (!(p.sa_flags & SA_SIGINFO) && p.sa_handler != SIG_DFL && p.sa_handler != SIG_IGN) return p.sa_handler(sig);
    signal(sig, SIG_DFL);
    raise(sig);
}

void on_kernel_fault(int sig, siginfo_t* info, void* uctx) {
    Recovery* r = t_recovery;
    if (!r || !t_in_kernel) return forward(sig, info, uctx);
    t_recovery = nullptr;
    r->fault.signal = sig;
    r->fault.address = uintptr_t(info->si_addr);
    r->fault.ip = uintptr_t(static_cast<ucontext_t*>(uctx)->uc_mcontext.gregs[REG_RIP]);
    siglongjmp(r->jump, 1);
}

// The job was cancelled (its program died): leave the kernel now. Outside
// kernel code the worker sees the flag at its next block.
void on_cancel(int, siginfo_t*, void*) {
    Recovery* r = t_recovery;
    if (!r || !t_in_kernel || !t_aborted || !t_aborted->load(std::memory_order_relaxed)) return;
    t_recovery = nullptr;
    r->fault.signal = kCancelSignal;
    siglongjmp(r->jump, 1);
}

void install_fault_handlers() {
    static std::once_flag once;
    std::call_once(once, [] {
        struct sigaction sa{};
        sa.sa_sigaction = on_kernel_fault;
        // On the worker's alternate stack: a runaway recursion faults on the
        // guard page of its own stack.
        sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
        for (int sig : {SIGSEGV, SIGBUS, SIGFPE, SIGILL}) sigaction(sig, &sa, &g_previous[sig]);
        struct sigaction cancel{};
        cancel.sa_sigaction = on_cancel;
        cancel.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESTART;
        sigaction(kCancelSignal, &cancel, nullptr);
    });
}

// The output of device printf for the job this worker runs.
thread_local int t_output_fd = 1;

}  // namespace

// ---- device printf
//
// cpusim-devc turns each printf of device code into calls of these, one per
// active lane: __cpusim_printf_begin(format) gives a record, one call per
// argument fills it, __cpusim_printf_end formats it and writes it, whole, to
// the program's standard output. Kernel modules resolve them here when they
// are loaded (the driver exports its symbols).
namespace {
struct PrintfArg {
    enum Kind { Int, Float, Pointer } kind;
    int bytes;
    union {
        int64_t i;
        double f;
        const void* p;
    };
};
struct PrintfRecord {
    const char* format = nullptr;
    std::vector<PrintfArg> args;
    bool used = false;
};
// The records of one worker: the lanes of a warp begin theirs before any ends.
thread_local std::vector<PrintfRecord> t_printf;

std::string format_record(const PrintfRecord& r) {
    std::string out;
    const char* f = r.format;
    size_t next = 0;
    char spec[64], buf[512];
    while (*f) {
        if (*f != '%') {
            out += *f++;
            continue;
        }
        if (f[1] == '%') {
            out += '%';
            f += 2;
            continue;
        }
        // %[flags][width][.precision][length]conversion, copied whole.
        const char* start = f++;
        while (*f && std::strchr("-+ #0123456789.*hlLzjt", *f)) ++f;
        if (!*f) break;
        const char conv = *f++;
        size_t len = std::min<size_t>(size_t(f - start), sizeof(spec) - 1);
        std::memcpy(spec, start, len);
        spec[len] = 0;
        if (next >= r.args.size()) {
            out += spec;  // more conversions than arguments: printed as written
            continue;
        }
        const PrintfArg& a = r.args[next++];
        int n = 0;
        if (std::strchr("diouxXc", conv)) {
            const bool wide = std::strstr(spec, "l") || std::strchr(spec, 'j') || std::strchr(spec, 'z');
            // Without a length modifier, the 32-bit value as passed.
            if (wide) n = std::snprintf(buf, sizeof buf, spec, (long long)a.i);
            else n = std::snprintf(buf, sizeof buf, spec, int(a.i));
        } else if (std::strchr("eEfFgGaA", conv)) {
            n = std::snprintf(buf, sizeof buf, spec, a.kind == PrintfArg::Float ? a.f : double(a.i));
        } else if (conv == 's') {
            n = std::snprintf(buf, sizeof buf, spec, a.p ? static_cast<const char*>(a.p) : "(null)");
        } else if (conv == 'p') {
            n = std::snprintf(buf, sizeof buf, spec, a.p);
        } else {
            n = std::snprintf(buf, sizeof buf, "%s", spec);
        }
        if (n > 0) out.append(buf, std::min<size_t>(size_t(n), sizeof buf - 1));
    }
    return out;
}
}  // namespace

extern "C" {
__attribute__((visibility("default"), used)) int64_t __cpusim_printf_begin(const char* format) {
    for (size_t i = 0; i < t_printf.size(); ++i)
        if (!t_printf[i].used) {
            t_printf[i] = {format, {}, true};
            return int64_t(i);
        }
    t_printf.push_back({format, {}, true});
    return int64_t(t_printf.size() - 1);
}
__attribute__((visibility("default"), used)) void __cpusim_printf_int(int64_t h, int64_t v, int32_t bytes) {
    PrintfArg a{PrintfArg::Int, bytes, {}};
    a.i = v;
    t_printf[size_t(h)].args.push_back(a);
}
__attribute__((visibility("default"), used)) void __cpusim_printf_float(int64_t h, double v) {
    PrintfArg a{PrintfArg::Float, 8, {}};
    a.f = v;
    t_printf[size_t(h)].args.push_back(a);
}
__attribute__((visibility("default"), used)) void __cpusim_printf_pointer(int64_t h, const void* v) {
    PrintfArg a{PrintfArg::Pointer, 8, {}};
    a.p = v;
    t_printf[size_t(h)].args.push_back(a);
}
__attribute__((visibility("default"), used)) int32_t __cpusim_printf_end(int64_t h) {
    PrintfRecord& r = t_printf[size_t(h)];
    std::string text = format_record(r);
    r.used = false;
    r.args.clear();
    // One write: lines from different threads never mix.
    for (size_t done = 0; done < text.size();) {
        ssize_t n = write(t_output_fd, text.data() + done, text.size() - done);
        if (n <= 0) break;
        done += size_t(n);
    }
    return int32_t(text.size());
}
}

Executor::Executor(const std::vector<int>& cpus) {
    // Pairs of SMT siblings, both among the workers (read from /sys, never
    // assumed from the numbering).
    pair_of_.assign(cpus.size(), -1);
    half_of_.assign(cpus.size(), 0);
    for (const PhysicalCore& core : read_topology().cores) {
        int found[2], n = 0;
        for (int cpu : core.cpus)
            for (size_t i = 0; i < cpus.size() && n < 2; ++i)
                if (cpus[i] == cpu) found[n++] = int(i);
        if (n < 2) continue;
        for (int h = 0; h < 2; ++h) {
            pair_of_[found[h]] = int(pairs_.size());
            half_of_[found[h]] = h;
        }
        pairs_.push_back(std::make_unique<Pair>());
    }
    install_fault_handlers();
    current_.reset(new std::atomic<Job*>[cpus.size()]);
    for (size_t i = 0; i < cpus.size(); ++i) current_[i] = nullptr;
    for (size_t i = 0; i < cpus.size(); ++i) threads_.emplace_back([this, i, cpu = cpus[i]] { worker(int(i), cpu); });
}

Executor::~Executor() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
        stopping_ = true;
    }
    start_cv_.notify_all();
    for (auto& t : threads_) t.join();
}

void Executor::worker(int index, int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);

    // Shared memory of the block this thread runs; first touched here, so it
    // lands in this core's NUMA node and stays hot in its caches.
    // Each worker's buffer starts at a different offset inside its page: the two
    // SMT siblings of a core share one L1, and page-aligned buffers would all
    // land in the same cache sets.
    auto* shared_mem = static_cast<uint8_t*>(std::aligned_alloc(4096, kMaxSharedBytes + 4096));
    uint8_t* shared = shared_mem + (uint64_t(index) * 1088) % 4096 / 64 * 64;
    CtxArray ctx;
    std::vector<uint8_t, AlignedAllocator> frames;
    std::vector<uint8_t> altstack(1 << 16);
    stack_t ss{};
    ss.ss_sp = altstack.data();
    ss.ss_size = altstack.size();
    sigaltstack(&ss, nullptr);

    while (true) {
        Job* job = nullptr;
        {
            // A job with blocks left, if the lock is free (when every worker
            // leaves a job at once, the others go on to wait for slots).
            std::unique_lock lock(mutex_, std::try_to_lock);
            if (lock.owns_lock()) {
                if (stop_) break;
                job = pick(index);
                lock.unlock();
            }
            if (!job) {
                // Nothing to do: spin a while, until a posted job wants
                // another worker; then sleep.
                bool slot = false;
                const auto until = std::chrono::steady_clock::now() + std::chrono::nanoseconds(cpusim::spin_ns());
                spinners_.fetch_add(1, std::memory_order_relaxed);
                // Parked on the demand's cache line where the CPU allows it:
                // a pause loop would slow down the SMT sibling still running
                // the last blocks of a job.
                const bool park = cpusim::has_mwaitx();
                for (unsigned i = 1; !stopping_.load(std::memory_order_relaxed); ++i) {
                    if (demand_.load(std::memory_order_relaxed) > 0 && take_demand()) {
                        slot = true;
                        break;
                    }
                    if (park) cpusim::park_on(&demand_, 4000);  // about a microsecond
                    else _mm_pause();
                    if ((park || i % 256 == 0) && std::chrono::steady_clock::now() > until) break;
                }
                spinners_.fetch_sub(1, std::memory_order_relaxed);
                if (slot) {
                    // The latest job, without the lock (usually the one that
                    // gave the slot).
                    readers_.fetch_add(1, std::memory_order_seq_cst);
                    Job* hot = hot_.load(std::memory_order_seq_cst);
                    if (hot && !hot->exhausted.load(std::memory_order_relaxed) && hot->try_join()) job = hot;
                    readers_.fetch_sub(1, std::memory_order_release);
                }
                if (!job) {
                    lock.lock();
                    if (stop_) break;
                    if (slot) {
                        job = pick(index);
                        if (!job) continue;  // its job needed no more workers
                    } else {
                        start_cv_.wait(lock, [&] { return stop_ || (job = pick(index)) != nullptr; });
                        if (!job) break;  // stopping
                    }
                    lock.unlock();
                }
            }
            if (job->cooperative) {
                std::lock_guard jl(mutex_);
                ++job->inflight;
                job->joined[size_t(index)] = 1;
                ++job->joins;
            }
        }
        alignas(64) uint64_t counters[CPUSIM_COUNTERS] = {};
        frames.resize(size_t(job->frame_stride) * job->warps->count);
        const auto t0 = std::chrono::steady_clock::now();
        Recovery recovery;
        t_aborted = &job->aborted;
        t_output_fd = job->cfg.output_fd;
        current_[size_t(index)].store(job, std::memory_order_release);
        if (sigsetjmp(recovery.jump, 1) == 0) {
            t_recovery = &recovery;
            if (!job->cooperative) run_blocks(*job, index, ctx, shared, counters, frames.data());
            else run_pair(*job, ctx, shared, counters, *pairs_[pair_of_[index]], half_of_[index],
                          pairs_[pair_of_[index]]->round[half_of_[index]]);
            t_recovery = nullptr;
        } else {
            // The kernel faulted, or its program is gone: the job stops. The
            // sibling of a pair must not wait for this half again.
            t_in_kernel = 0;
            job->aborted.store(true, std::memory_order_relaxed);
            // No block will be run any more: say so, or this worker, if it
            // is the last to leave, never closes the job (leave() needs it).
            job->exhausted.store(true, std::memory_order_release);
            if (job->cooperative) pairs_[pair_of_[index]]->epoch(half_of_[index]).store(UINT64_MAX);
            std::lock_guard lock(mutex_);
            if (!job->fault.signal) job->fault = recovery.fault;
        }
        const uint64_t ns = uint64_t(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        current_[size_t(index)].store(nullptr, std::memory_order_release);
        job->busy_ns.fetch_add(ns, std::memory_order_relaxed);
        if (job->counters) {  // profiling runs only
            std::lock_guard lock(mutex_);
            for (int i = 0; i < CPUSIM_COUNTERS; ++i) job->counters[i] += counters[i];
        }
        if (job->cooperative) {
            std::lock_guard lock(mutex_);
            --job->inflight;
            maybe_finish(job);
        } else if (job->leave()) {
            std::lock_guard lock(mutex_);
            finish(job);
        }
    }
    std::free(shared_mem);
}

bool Executor::Job::has_work() const {
    if (aborted.load(std::memory_order_relaxed)) return false;
    if (!ranges) return next.load(std::memory_order_relaxed) < blocks;
    for (size_t i = 0; i < nranges; ++i)
        if (ranges[i].next.load(std::memory_order_relaxed) < ranges[i].end) return true;
    return false;
}

// The next block for `worker`: from its own range, then from the others'
// (starting with its neighbours, which are on the same CCD more often).
bool Executor::claim(const Job& job, int worker, uint64_t* b) {
    if (job.exhausted.load(std::memory_order_relaxed)) return false;
    if (job.aborted.load(std::memory_order_relaxed)) {
        job.exhausted.store(true, std::memory_order_release);
        return false;
    }
    for (size_t k = 0; k < job.nranges; ++k) {
        const size_t w = (size_t(worker) + k) % job.nranges;
        const Job::Range& r = job.ranges[w];
        if (r.next.load(std::memory_order_relaxed) >= r.end) continue;
        uint64_t t = r.next.fetch_add(1, std::memory_order_relaxed);
        if (t < r.end) {
            *b = job.block_at(w, t);
            return true;
        }
    }
    job.exhausted.store(true, std::memory_order_release);
    return false;
}

void Executor::run_blocks(const Job& job, int worker, CtxArray& ctx, uint8_t* shared, uint64_t* counters,
                          uint8_t* frames) {
    ctx = *job.warps;
    const size_t n = ctx.count;
    for (size_t w = 0; w < n; ++w) {
        cpusim_warp_ctx* c = ctx.at(w);
        c->params = job.params;
        c->shared = shared;
        c->counters = counters;
        c->frame = job.frame_stride ? frames + w * job.frame_stride : nullptr;
    }
    std::vector<uint8_t> done(n);
    std::vector<uint32_t> phase(n);
    if (job.split > 1) {
        // One group of a block per unit; the groups need nothing else.
        uint64_t u;
        while (claim(job, worker, &u)) {
            cpusim_warp_ctx* c = ctx.at(size_t(u % job.split));
            block_coords(job, u / job.split, c->block_idx);
            in_kernel(job.entry, c);
        }
        return;
    }
    uint64_t b;
    while (claim(job, worker, &b)) {
        uint32_t idx[3];
        block_coords(job, b, idx);
        const uint32_t bx = idx[0], by = idx[1], bz = idx[2];
        for (size_t w = 0; w < n; ++w) {
            cpusim_warp_ctx* c = ctx.at(w);
            c->block_idx[0] = bx;
            c->block_idx[1] = by;
            c->block_idx[2] = bz;
        }
        if (!job.frame_stride) {
            for (size_t w = 0; w < n; ++w) in_kernel(job.entry, ctx.at(w));
            continue;
        }
        // Barriers: each call runs a warp up to its next __syncthreads (or its
        // end). A warp's state 0 means "from the start".
        for (size_t w = 0; w < n; ++w) *reinterpret_cast<uint32_t*>(ctx.at(w)->frame) = 0;
        std::fill(done.begin(), done.end(), 0);
        if (job.block) {
            t_in_kernel = 1;
            job.block(ctx.at(0), uint32_t(n), done.data());
            t_in_kernel = 0;
            continue;
        }
        // Always run a warp that is behind (the lowest barrier count). When
        // it is the last one behind, every other warp already waits at its
        // next barrier, so it may go through that barrier without suspending.
        std::fill(phase.begin(), phase.end(), 0);
        size_t remaining = n;
        while (remaining) {
            uint32_t low = UINT32_MAX;
            size_t behind = 0;
            for (size_t w = 0; w < n; ++w) {
                if (done[w]) continue;
                if (phase[w] < low) low = phase[w], behind = 0;
                if (phase[w] == low) ++behind;
            }
            for (size_t w = 0; w < n && behind; ++w) {
                if (done[w] || phase[w] != low) continue;
                cpusim_warp_ctx* c = ctx.at(w);
                static const bool disabled = std::getenv("CPUSIM_NO_BARRIER_PASS") != nullptr;  // for comparisons
                // (Not with yield points: a warp could pass a barrier, then
                // yield, and its count would be ambiguous.)
                const bool pass = behind == 1 && !disabled && !(job.meta_flags & CPUSIM_META_YIELDS);
                c->barrier_pass = pass;
                --behind;
                const uint32_t r = in_kernel(job.entry, c);
                if (r == CPUSIM_EXIT) {
                    done[w] = 1;
                    --remaining;
                } else if (r != CPUSIM_YIELD) {
                    phase[w] += pass ? 2 : 1;  // went through one barrier, stopped at the next
                }  // a yield: the same count; the warps waiting with it run first
            }
        }
    }
}

void Executor::block_coords(const Job& job, uint64_t b, uint32_t* idx) const {
    const uint32_t gx = job.cfg.grid[0], gy = job.cfg.grid[1];
    // Block order (CUDA promises none): for 2D grids, bands of kBand columns
    // walked row by row, so the blocks running at the same time touch a few
    // rows and a few columns of the data instead of one row and many columns
    // (the classic thread-block swizzle; it keeps both operands of a matmul in
    // cache). CPUSIM_BLOCK_ORDER=linear disables it.
    static const bool linear = [] {
        const char* v = std::getenv("CPUSIM_BLOCK_ORDER");
        return v && std::strcmp(v, "linear") == 0;
    }();
    constexpr uint32_t kBand = 8;
    const bool swizzle = !linear && gy > 1 && gx > kBand;
    uint64_t plane = uint64_t(gx) * gy;
    idx[2] = uint32_t(b / plane);
    uint64_t r = b % plane;
    if (swizzle) {
        uint64_t band = r / (uint64_t(kBand) * gy);       // which band of columns
        uint32_t band_x0 = uint32_t(band * kBand);
        uint32_t width = std::min(kBand, gx - band_x0);    // the last band may be narrower
        uint64_t in_band = r - band * uint64_t(kBand) * gy;
        idx[0] = band_x0 + uint32_t(in_band % width);
        idx[1] = uint32_t(in_band / width);
    } else {
        idx[0] = uint32_t(r % gx);
        idx[1] = uint32_t(r / gx);
    }
}

// One worker of a pair. Per block (round r), the first worker takes the next
// block and hands it over in slot[r & 1]; each runs its half (warp_index 0 or
// lanes / 32) in the first worker's shared memory. Epochs order everything:
//   - a half sets its epoch to r << 32 when it starts block r, and the code's
//     barriers count from there (kernel_abi.h);
//   - when it returns, it sets (r << 32) | 0xffffffff: the other half's
//     barriers no longer wait for it, and the first worker knows the shared
//     memory is free for the next block.
// The second worker starts block r only after the first one published it,
// which it does only after finishing block r - 1 itself.
void Executor::run_pair(const Job& job, CtxArray& ctx, uint8_t* shared, uint64_t* counters, Pair& pair, int half,
                        uint64_t& round) {
    ctx = *job.warps;
    cpusim_warp_ctx* c = ctx.at(size_t(half));
    c->params = job.params;
    c->counters = counters;
    c->frame = nullptr;
    c->coop_epoch = reinterpret_cast<uint64_t*>(&pair.epoch(half));
    c->coop_other = reinterpret_cast<const uint64_t*>(&pair.epoch(1 - half));
    std::atomic<uint64_t>& mine = pair.epoch(half);
    std::atomic<uint64_t>& other = pair.epoch(1 - half);
    while (true) {
        const uint64_t r = ++round;
        Pair::Slot& slot = pair.slot[r & 1];
        uint64_t b;
        if (half == 0) {
            // The other half left block r - 1 (and the slot's last reader is done).
            while (other.load(std::memory_order_acquire) < ((r - 1) << 32 | 0xffffffffu)) _mm_pause();
            b = job.aborted.load(std::memory_order_relaxed) ? job.blocks : job.next.fetch_add(1, std::memory_order_relaxed);
            slot.block = b;
            slot.shared = shared;
            slot.round.store(r, std::memory_order_release);
        } else {
            while (slot.round.load(std::memory_order_acquire) != r) {
                if (job.aborted.load(std::memory_order_relaxed)) {  // the first half faulted
                    mine.store(UINT64_MAX, std::memory_order_release);
                    return;
                }
                _mm_pause();
            }
            b = slot.block;
        }
        if (b >= job.blocks) {
            mine.store(r << 32 | 0xffffffffu, std::memory_order_release);
            return;
        }
        c->shared = slot.shared;
        block_coords(job, b, c->block_idx);
        mine.store(r << 32, std::memory_order_release);
        in_kernel(job.entry, c);
        mine.store(r << 32 | 0xffffffffu, std::memory_order_release);
    }
}

Executor::Job* Executor::pick(int worker) {
    for (Job* j : active_) {
        if (j->cooperative) {
            // Every worker with a sibling takes part exactly once (the pair
            // hands blocks over and counts rounds together).
            if (pair_of_[size_t(worker)] >= 0 && !j->joined[size_t(worker)]) return j;
            continue;
        }
        if (!j->exhausted.load(std::memory_order_relaxed) && !j->aborted.load(std::memory_order_relaxed) &&
            j->try_join())
            return j;
    }
    return nullptr;
}

void Executor::cancel(uint32_t tag) {
    std::lock_guard lock(mutex_);
    for (Job* j : active_) {
        if (tag && j->tag != tag) continue;
        j->aborted.store(true, std::memory_order_relaxed);
        if (!j->fault.signal) j->fault.signal = kCancelSignal;
        for (size_t i = 0; i < threads_.size(); ++i)
            if (current_[i].load(std::memory_order_acquire) == j) pthread_kill(threads_[i].native_handle(), kCancelSignal);
    }
}

void Executor::finish(Job* j) {
    active_.erase(std::find(active_.begin(), active_.end(), j));
    j->done.store(true, std::memory_order_release);  // last: the waiting run() may return at once
    done_cv_.notify_all();
}

// Cooperative jobs: over when every paired worker joined and left.
void Executor::maybe_finish(Job* j) {
    if (j->done || j->inflight > 0) return;
    if (j->joins < int(2 * pairs_.size())) return;
    if (j->cooperative && j->aborted.load(std::memory_order_relaxed))
        for (auto& p : pairs_) p->resync();
    active_.erase(std::find(active_.begin(), active_.end(), j));
    j->done.store(true, std::memory_order_release);  // last: the waiting run() may return at once
    done_cv_.notify_all();
}

CtxArray Executor::make_groups(const LaunchConfig& cfg, uint32_t lanes, uint32_t warp_size) {
    // Lane -> thread index; the same for all blocks.
    const uint32_t bx = cfg.block[0], by = cfg.block[1], bz = cfg.block[2];
    const uint32_t threads = bx * by * bz;
    CtxArray groups;
    groups.count = (threads + lanes - 1) / lanes;
    groups.stride = cpusim_stagger(cpusim_ctx_size(lanes));
    groups.data.assign(size_t(groups.stride) * groups.count, 0);
    for (uint32_t g = 0; g < groups.count; ++g) {
        cpusim_warp_ctx* c = groups.at(g);
        c->lanes = lanes;
        c->warp_index = g * lanes / warp_size;  // 0 for a block smaller than a warp
        uint32_t* tx = cpusim_ctx_tid(c, 0);
        uint32_t* ty = cpusim_ctx_tid(c, 1);
        uint32_t* tz = cpusim_ctx_tid(c, 2);
        uint32_t* mask = cpusim_ctx_lane_mask(c);
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            uint32_t t = g * lanes + lane;
            if (t >= threads) break;
            tx[lane] = t % bx;
            ty[lane] = t / bx % by;
            tz[lane] = t / (bx * by);
            mask[lane / 32] |= 1u << (lane % 32);
        }
        std::memcpy(c->block_dim, cfg.block, sizeof(c->block_dim));
        std::memcpy(c->grid_dim, cfg.grid, sizeof(c->grid_dim));
        c->huge_lo = cfg.huge_lo;
        c->huge_hi = cfg.huge_hi;
    }
    return groups;
}

void Executor::run(cpusim_entry_fn entry, uint32_t lanes, bool suspending, const cpusim_kernel_meta& meta,
                   const LaunchConfig& cfg, const uint8_t* params, uint64_t* counters, cpusim_block_fn block,
                   bool cooperative, uint64_t first, uint64_t last, uint64_t* busy_ns, KernelFault* fault,
                   uint32_t tag) {
    CtxArray warps = make_groups(cfg, lanes, meta.warp_size);
    // Block-wide entries never suspend: no frames.
    uint32_t frame_bytes = 0;
    if (suspending)
        frame_bytes = lanes == meta.warp_lanes ? meta.frame_bytes : meta.frame_bytes_wide[lanes == 2 * meta.warp_size ? 0 : 1];

    const uint64_t total = uint64_t(cfg.grid[0]) * cfg.grid[1] * cfg.grid[2];
    Job job;
    job.entry = entry;
    job.cfg = cfg;
    job.params = params;
    job.warps = &warps;
    job.blocks = std::min(total, last);
    job.counters = counters;
    job.frame_stride = frame_bytes ? cpusim_stagger(frame_bytes) : 0;
    job.block = block;
    job.cooperative = cooperative && !pairs_.empty();
    job.tag = tag;
    job.meta_flags = meta.flags;
    // The warps of a block that has no barrier and no shared memory never
    // talk to each other: with few blocks, hand out its groups one by one
    // (as a GPU schedules warps), so the work spreads over every worker
    // instead of one worker per block. matrix-rotate launches 10 blocks.
    if (!job.cooperative && !block && !(meta.flags & CPUSIM_META_HAS_BARRIERS) && cfg.shared_bytes == 0 &&
        warps.count > 1 && total < 8 * threads_.size()) {
        job.split = warps.count;
        job.blocks *= job.split;
        first *= job.split;
    }
    job.next.store(first, std::memory_order_relaxed);
    if (counters) std::memset(counters, 0, sizeof(uint64_t) * CPUSIM_COUNTERS);
    if (busy_ns) *busy_ns = 0;
    if (first >= job.blocks) return;
    if (!job.cooperative) {
        // Worker w's sequence: chunks w, w + n, w + 2n... of `chunk` blocks.
        job.nranges = threads_.size();
        job.ranges.reset(new Job::Range[job.nranges]);
        job.first = first;
        const uint64_t count = job.blocks - first;
        job.chunk = std::clamp<uint64_t>(count / job.nranges, 1, Job::kChunk);
        const uint64_t per_round = job.chunk * job.nranges;
        for (size_t w = 0; w < job.nranges; ++w) {
            const uint64_t full = count / per_round * job.chunk;  // whole rounds
            const uint64_t rest = count % per_round, start = w * job.chunk;
            job.ranges[w].end = full + (rest > start ? std::min(job.chunk, rest - start) : 0);
        }
    }

    std::unique_lock lock(mutex_);
    if (job.cooperative) {
        done_cv_.wait(lock, [&] { return active_.empty() && !coop_active_; });
        coop_active_ = true;
        job.joined.assign(threads_.size(), 0);
    } else {
        done_cv_.wait(lock, [&] { return !coop_active_; });
    }
    active_.push_back(&job);
    if (!job.cooperative) hot_.store(&job, std::memory_order_release);
    posted_.fetch_add(1, std::memory_order_release);
    {
        const int workers = int(threads_.size());
        const uint64_t units = job.cooperative ? uint64_t(workers) : job.blocks - first;
        const int want = int(std::min<uint64_t>(units, uint64_t(workers)));
        if (demand_.fetch_add(want, std::memory_order_release) + want > workers)
            demand_.store(workers, std::memory_order_relaxed);  // stale demand of finished jobs
        // Wake sleepers only if the idle workers spinning are too few: a
        // one-block kernel must not wake every idle worker. (One that stops
        // spinning meanwhile finds the job when it goes to sleep: it looks
        // for one under the lock first, and the job is already listed.)
        if (job.cooperative || want > spinners_.load(std::memory_order_relaxed)) start_cv_.notify_all();
    }
    // Short kernels end within the spin: no sleep, no wake-up.
    lock.unlock();
    // (This thread is not pinned: the short budget, common/spin.h.)
    cpusim::spin_until([&] { return job.done.load(std::memory_order_acquire); });
    lock.lock();
    done_cv_.wait(lock, [&] { return job.done.load(); });
    // No worker may still be looking at the job when it goes away (it lives
    // on this stack). Both sides store, then load the other's variable: the
    // orderings must be sequentially consistent, or x86's store buffer lets
    // each miss the other's store (we see no reader, the worker sees the job).
    Job* expected = &job;
    hot_.compare_exchange_strong(expected, nullptr, std::memory_order_seq_cst);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    while (readers_.load(std::memory_order_seq_cst)) _mm_pause();
    if (busy_ns) *busy_ns = job.busy_ns;
    if (fault) *fault = job.fault;
    if (job.cooperative) {
        coop_active_ = false;
        done_cv_.notify_all();
    }
}

}  // namespace cpusim
