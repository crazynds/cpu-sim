#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <unordered_map>
#include <vector>

#include "device/kernel_abi.h"

namespace cpusim {

// 64-byte aligned storage (contexts, warp frames: the generated code assumes it).
template <class T>
struct AlignedAlloc {
    using value_type = T;
    AlignedAlloc() = default;
    template <class U>
    AlignedAlloc(const AlignedAlloc<U>&) {}
    T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(64))); }
    void deallocate(T* p, size_t) { ::operator delete(p, std::align_val_t(64)); }
    template <class U>
    bool operator==(const AlignedAlloc<U>&) const { return true; }
};

// The contexts of one block's calls, back to back (cpusim_ctx_size apart).
struct CtxArray {
    std::vector<uint8_t, AlignedAlloc<uint8_t>> data;
    uint32_t stride = 0, count = 0;
    cpusim_warp_ctx* at(size_t i) { return reinterpret_cast<cpusim_warp_ctx*>(data.data() + i * stride); }
};

struct LaunchConfig {
    uint32_t grid[3] = {1, 1, 1};
    uint32_t block[3] = {1, 1, 1};
    uint32_t shared_bytes = 0;
    bool profile = false;  // run the instrumented entry and collect counters
    int debug = 0;         // 1: single-step with guard pages (debug_exec.h); 2: also trace every step
    int output_fd = 1;     // where the kernel's printf writes (the program's stdout)
    // Span of the objects of 2 GiB or more (cpusim_warp_ctx::huge_lo/hi);
    // by default everything, so 64-bit gather indices everywhere.
    uint64_t huge_lo = 0, huge_hi = UINT64_MAX;
};

struct LaunchResult {
    uint64_t kernel_ns = 0;                    // wall time of the kernel itself
    uint64_t counters[CPUSIM_COUNTERS] = {};   // filled when profiling
    char report[1024] = {};                    // debug mode: what went wrong
};

// The device's "SMs": one worker thread per reserved logical CPU, pinned to it.
// A launch hands out whole blocks through an atomic counter; each worker runs
// every warp of its block. With a cooperative entry, the two SMT siblings of a
// core run one block together instead, half each (DESIGN.md 4.2).
// Several launches (from different streams) can be active at once: a worker
// takes blocks from the oldest one that still has some, so a kernel's last
// blocks overlap with the next kernel's first ones. Cooperative launches run
// alone (every pair of siblings has to take part).
// A kernel that faulted (SIGSEGV, SIGBUS, SIGFPE or SIGILL on a worker): the
// launch stops, and only it fails; the process (the driver, which serves
// every program) goes on.
struct KernelFault {
    int signal = 0;  // 0: none
    uintptr_t address = 0;
    uintptr_t ip = 0;
};

class Executor {
public:
    explicit Executor(const std::vector<int>& cpus);
    ~Executor();

    // Runs the kernel to completion. `params` must stay valid until it returns.
    // With `counters`, every worker counts into its own slots and the sums land there.
    // `meta` tells the per-warp frame size; cfg.shared_bytes is the block's total
    // shared memory (static + dynamic).
    // `lanes`: threads per entry call (32, or a block-wide variant's width).
    // `suspending`: the entry may return CPUSIM_BARRIER (needs frames).
    // `block`: when given (32-lane path), runs each block's barrier rounds in one call.
    // `cooperative`: a __cpusim_entry<lanes>c_ entry; the block is two calls of
    // `lanes`, run by the two SMT siblings of a core at the same time.
    // [first, last): only these blocks, in hand-out order (the width tuning
    // runs slices of one launch with different entries).
    void run(cpusim_entry_fn entry, uint32_t lanes, bool suspending, const cpusim_kernel_meta& meta,
             const LaunchConfig& cfg, const uint8_t* params, uint64_t* counters = nullptr,
             cpusim_block_fn block = nullptr, bool cooperative = false, uint64_t first = 0,
             uint64_t last = UINT64_MAX, uint64_t* busy_ns = nullptr, KernelFault* fault = nullptr,
             uint32_t tag = 0);
    // Stops the running launches tagged `tag` (0: all): their workers leave
    // the kernel code they are in. Their run() returns with a fault of
    // SIGUSR2. Call again while they last: a worker outside kernel code
    // notices at its next block.
    void cancel(uint32_t tag);
    // busy_ns: the time workers spent running this launch's blocks, summed
    // over workers. Divided by the blocks, the cost of a block: unlike wall
    // time, it ignores the uneven end of a launch and the launch overhead,
    // and other launches running at the same time (streams).

    // Per-call contexts for one block: thread index of every lane, lane masks.
    static CtxArray make_groups(const LaunchConfig& cfg, uint32_t lanes, uint32_t warp_size);

    size_t workers() const { return threads_.size(); }
    // Pairs of SMT siblings among the workers (0: no cooperative launches).
    size_t pairs() const { return pairs_.size(); }

    static constexpr uint32_t kMaxSharedBytes = 100 * 1024;

private:
    struct Job {
        cpusim_entry_fn entry = nullptr;
        LaunchConfig cfg;
        const uint8_t* params = nullptr;
        const CtxArray* warps = nullptr;  // one template per call of a block
        uint64_t blocks = 0;              // hand out blocks up to this one
        uint64_t* counters = nullptr;
        uint32_t frame_stride = 0;  // bytes per warp frame, 0 without barriers
        cpusim_block_fn block = nullptr;
        bool cooperative = false;
        // > 1: the unit handed out is one group of a block, not the block
        // (kernels without barriers or shared memory, small grids): `next`
        // and `blocks` count units, unit u = block u / split, group u % split.
        uint32_t split = 1;
        mutable std::atomic<uint64_t> next{0};  // cooperative jobs: next block to hand out
        // Everything else: blocks (units) dealt out in chunks of kChunk, chunk
        // c to worker c % workers, each walking its chunks in order; a worker
        // done with its own takes from the others'. Every worker moves
        // through the grid in step with the rest (neighbouring blocks run at
        // the same time and share what they read, as with one shared
        // counter), and the same blocks land on the same core launch after
        // launch, so a kernel run again on the same data (stencil time steps)
        // finds its part in that core's caches.
        // 64: stencils keep long runs of neighbouring blocks (iso2dfd: 0.41 ms
        // against 0.51 with 4), a matmul still shares its operands (~5% more
        // cycles than 4).
        static constexpr uint64_t kChunk = 64;
        // This job's chunk: kChunk, or less when the grid is too small to
        // give every worker a whole one (1024 blocks were 16 chunks for 28
        // workers: a dozen of them stole block by block from the rest).
        uint64_t chunk = kChunk;
        struct alignas(64) Range {
            mutable std::atomic<uint64_t> next{0};  // position in this worker's own sequence
            uint64_t end = 0;                       // its length
        };
        std::unique_ptr<Range[]> ranges;
        size_t nranges = 0;
        uint64_t first = 0;
        // Block (unit) at position t of worker w's sequence.
        uint64_t block_at(size_t w, uint64_t t) const {
            return first + ((t / chunk) * nranges + w) * chunk + t % chunk;
        }
        bool has_work() const;  // some block not taken yet (reads every range: not on hot paths)
        // A worker found no block left: no one else needs to join.
        mutable std::atomic<bool> exhausted{false};
        // Cooperative jobs, under mutex_: workers inside, and which have joined.
        int inflight = 0;
        // Other jobs, without the lock: workers inside, plus kClosed once the
        // last one left with no block to hand out (then nobody can join).
        static constexpr int64_t kClosed = int64_t(1) << 62;
        std::atomic<int64_t> state{0};
        // At most this many workers inside at once (Executor::cap_for).
        int64_t cap = INT64_MAX;
        bool try_join() {
            int64_t v = state.load(std::memory_order_relaxed);
            while (!(v & kClosed) && v < cap)
                if (state.compare_exchange_weak(v, v + 1, std::memory_order_acquire)) return true;
            return false;
        }
        // True for the worker that closes the job.
        bool leave() {
            if (state.fetch_sub(1, std::memory_order_acq_rel) != 1 || !exhausted.load(std::memory_order_acquire))
                return false;
            int64_t zero = 0;
            return state.compare_exchange_strong(zero, kClosed, std::memory_order_acq_rel);
        }
        int joins = 0;
        std::atomic<uint64_t> busy_ns{0};
        std::vector<char> joined;
        std::atomic<bool> done{false};
        // A worker faulted: the others take no more blocks.
        mutable std::atomic<bool> aborted{false};
        KernelFault fault;  // under mutex_
        uint32_t tag = 0;   // the context that launched it
        uint32_t meta_flags = 0;  // CPUSIM_META_*
    };

    // Two SMT siblings running blocks together. Barrier epochs (see
    // kernel_abi.h) are (round << 32) | barriers passed, one cache line each;
    // round r's block goes from the first worker of the pair to the second
    // through slot[r & 1].
    struct alignas(64) Pair {
        alignas(64) std::atomic<uint64_t> epoch0{0xffffffffu};
        alignas(64) std::atomic<uint64_t> epoch1{0xffffffffu};
        struct alignas(64) Slot {
            std::atomic<uint64_t> round{0};
            uint64_t block = 0;
            uint8_t* shared = nullptr;
        } slot[2];
        std::atomic<uint64_t>& epoch(int half) { return half ? epoch1 : epoch0; }
        // Blocks each half went through (equal between jobs; written only by
        // that half's worker).
        uint64_t round[2] = {0, 0};
        // After a fault ended a job midway: both halves agree again.
        void resync() {
            const uint64_t r = std::max(round[0], round[1]);
            round[0] = round[1] = r;
            epoch0.store(r << 32 | 0xffffffffu);
            epoch1.store(r << 32 | 0xffffffffu);
        }
    };

    // Workers worth calling for a launch of `units` blocks (or groups) of a
    // kernel whose cost per unit was measured before: about CPUSIM_TARGET_US
    // (2 us) of work each. Every worker that joins a job costs a few atomic operations
    // on lines all the others touch too, serialized, and across sockets on a
    // two-socket machine: with 124 workers a 1024-block kernel of 50 us of
    // work took 170 us, nearly all of it joining and leaving.
    int cap_for(uint64_t key, uint64_t units);
    void learn_cost(uint64_t key, uint64_t units, uint64_t busy_ns);
    std::unordered_map<uint64_t, double> cost_ns_;  // per (entry, width, split): ns per unit, under mutex_
    void worker(int index, int cpu);
    Job* pick(int worker);    // under mutex_: a job this worker should take part in, or null
    void maybe_finish(Job* job);  // under mutex_
    void run_blocks(const Job& job, int worker, CtxArray& ctx, uint8_t* shared, uint64_t* counters, uint8_t* frames);
    static bool claim(const Job& job, int worker, uint64_t* b);
    void run_pair(const Job& job, CtxArray& ctx, uint8_t* shared, uint64_t* counters, Pair& pair, int half,
                  uint64_t& round);
    // blockIdx of the b-th block handed out (CUDA promises no order).
    void block_coords(const Job& job, uint64_t b, uint32_t* idx) const;

    std::vector<std::thread> threads_;
    std::vector<std::unique_ptr<Pair>> pairs_;
    std::vector<int> pair_of_, half_of_;  // per worker; pair -1: no sibling among the workers
    // A plain mutex: the hot paths (joining and leaving a job) take no lock,
    // and a spin lock whose holder gets preempted (an unpinned thread on its
    // CPU) stalls every worker waiting for it.
    std::mutex mutex_;
    std::condition_variable start_cv_, done_cv_;
    bool stop_ = false;
    // Idle workers and a launching thread spin a while before sleeping: a
    // kernel launched right after another finds the workers awake, with no
    // system call on either side (waking 28 sleeping threads per launch cost
    // tens of microseconds). Jobs posted so far, to notice a new one.
    std::atomic<uint64_t> posted_{0};
    // Workers a posted job can use (its blocks, at most every worker): an
    // idle worker takes one before it takes the lock, so a one-block kernel
    // does not have every worker queue on the mutex.
    std::atomic<int> demand_{0};
    std::atomic<int> spinners_{0};  // idle workers spinning for a demand slot
    static constexpr int kWakeFanout = 2;  // sleepers each wake-up passes on (worker())
    bool take_demand() {
        int d = demand_.load(std::memory_order_relaxed);
        while (d > 0)
            if (demand_.compare_exchange_weak(d, d - 1, std::memory_order_acquire)) return true;
        return false;
    }
    std::atomic<bool> stopping_{false};
    std::vector<Job*> active_;   // oldest first
    std::unique_ptr<std::atomic<Job*>[]> current_;  // the job each worker is in (for cancel)
    // The latest job posted (not cooperative): a worker with a demand slot
    // joins it without the lock. `readers_` counts workers between reading it
    // and joining, so the job outlives their look at it.
    std::atomic<Job*> hot_{nullptr};
    std::atomic<int> readers_{0};
    void finish(Job* job);  // under mutex_: the job is over, run() may return
    bool coop_active_ = false;   // a cooperative job runs alone
};

}  // namespace cpusim
