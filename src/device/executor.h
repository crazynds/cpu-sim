#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
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
};

struct LaunchResult {
    uint64_t kernel_ns = 0;                    // wall time of the kernel itself
    uint64_t counters[CPUSIM_COUNTERS] = {};   // filled when profiling
    char report[1024] = {};                    // debug mode: what went wrong
};

// The device's "SMs": one worker thread per reserved logical CPU, pinned to it.
// A launch hands out whole blocks through an atomic counter; each worker runs
// every warp of its block. With a cooperative entry, the two SMT siblings of a
// core run one block together instead, half each (DESIGN.md 4.3).
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
    static CtxArray make_groups(const LaunchConfig& cfg, uint32_t lanes);

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
        struct alignas(64) Range {
            mutable std::atomic<uint64_t> next{0};  // position in this worker's own sequence
            uint64_t end = 0;                       // its length
        };
        std::unique_ptr<Range[]> ranges;
        size_t nranges = 0;
        uint64_t first = 0;
        // Block (unit) at position t of worker w's sequence.
        uint64_t block_at(size_t w, uint64_t t) const {
            return first + ((t / kChunk) * nranges + w) * kChunk + t % kChunk;
        }
        bool has_work() const;  // some block not taken yet
        // Under mutex_: workers inside the job, and (cooperative) which have joined.
        int inflight = 0;
        int joins = 0;
        uint64_t busy_ns = 0;
        std::vector<char> joined;
        bool done = false;
        // A worker faulted: the others take no more blocks.
        mutable std::atomic<bool> aborted{false};
        KernelFault fault;  // under mutex_
        uint32_t tag = 0;   // the context that launched it
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
    std::mutex mutex_;
    std::condition_variable start_cv_, done_cv_;
    bool stop_ = false;
    std::vector<Job*> active_;   // oldest first
    std::vector<Job*> current_;  // under mutex_: the job each worker is in
    bool coop_active_ = false;   // a cooperative job runs alone
};

}  // namespace cpusim
