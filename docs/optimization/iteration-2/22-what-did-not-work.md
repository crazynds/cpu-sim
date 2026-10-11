# 22. What did not work

> **In one sentence:** a record of the project's negative attempts: ideas that looked good, were tested, and made performance worse, changed nothing, or measured the wrong thing; each item says what was expected, what happened and why, so that they are not repeated without that context.

The project's general concepts are in [00-basic-concepts.md](../00-basic-concepts.md). The document for each optimization explains the context in detail; here only the essentials of each attempt are kept.

> **Iteration:** gathers attempts from both iterations. Those from iteration 1 are also summarized in the [original record](../iteration-1/OPTIMIZATIONS.md).

## The problem

In performance optimization, many plausible hypotheses are not confirmed: a change that should help may make things worse through a second-order effect (register pressure, cache behavior, measurement noise). Without a record, the same idea comes back later, consumes the same time and reaches the same result.

There is also an error of a different nature: **measuring wrong**. Since the width choice is automatic, a poorly chosen metric makes the system itself take the wrong decision, and the result looks like an optimization that did not work when the defect was in the metric.

## The idea

For each attempt, record three things:

1. **What was expected** (the hypothesis).
2. **What happened** (the measurement or observation).
3. **Why** (the explanation, when it was found).

And indicate whether the idea was **removed**, left **off by default** (it still exists and can be turned on for comparisons) or remains as a **candidate** that the system sometimes chooses.

## How it works, step by step

The list below follows the approximate order in which things were tried.

### 1. Block function (`__cpusim_block_<kernel>`)

**State:** off by default; `CPUSIM_BLOCK_FN=1` turns it on.

**What was expected.** In a kernel with barriers, the executor calls the function of each warp up to the next barrier, then that of the next warp, and so on, round after round ([09-state-at-barriers.md](../iteration-1/09-state-at-barriers.md)). Each call has a cost (saving and restoring state, returning to the executor). The idea was to generate a single function per block, with the barrier rounds **inside** the generated code and the warp code copied into it (inlined), eliminating the calls.

**What happened.** It became **slower** than the per-warp calls.

**Why.** Register pressure. With the warp body inlined inside the round loop, the live values of both compete for the 32 zmm registers (and for the 8 mask registers `k`); the register allocator starts generating 64-byte spills and reloads on the stack inside the hot loop, and this traffic costs more than the calls saved. In addition, it does not take advantage of the direct pass through the barrier by the last warp (`barrier_pass`, see [10-barrier-pass.md](10-barrier-pass.md)).

**Note.** The old record in `OPTIMIZATIONS.md` says that `CPUSIM_NO_BLOCK_FN=1` turns off the block function. In the current code the sense is the opposite: it already comes off and `CPUSIM_BLOCK_FN` turns it on (excerpt in the "The code" section).

### 2. Whole-block variant without JIT

**State:** the variant exists; without JIT it loses, with JIT it wins on some kernels.

**What was expected.** Instead of running a block of 256 threads as 8 calls of 32 lanes, run it as one call of 256 lanes ([02-width-variants.md](../iteration-1/02-width-variants.md)). Fewer calls, fewer suspensions at barriers.

**What happened.** Compiled generically (without knowing the block size at compile time), it was **worse than the 32-lane one**.

**Why.** With 256 lanes, each memory access involves 256 pointers of 8 bytes, that is, 32 zmm registers for addresses alone, and without knowing the block size the code needs extra tests for lanes that do not exist. Added to the other live values, this exceeds the 32 zmm, and the effect is the same as in item 1: spills in the hot loop. With the specialized JIT ([01-specialized-jit.md](../iteration-1/01-specialized-jit.md)), which knows the block size and simplifies the addresses, this variant became **the best** for tiled matmul. The idea was not wrong; the specialization was missing.

### 3. `mmap` with flags or prefetch for shared memory

**State:** not implemented (the hypothesis was discarded by measurement).

**What was expected.** The hypothesis (from the user) was that shared memory was slow because of how the memory was obtained from the system, and that mapping it with special `mmap` flags or prefetching it would solve it.

**What happened.** Measuring, shared memory was **already in the L1 cache** (it is small and each worker reuses the same area all the time). The actual accesses to shared were about **7% of the cycles**. Even if they became free, the maximum gain would be small.

**Why.** The slow part was not the memory: it was the **generated code** around the accesses (address computation, gathers where a contiguous read would do). This was solved by the static access analysis, described in [12-static-access-analysis.md](12-static-access-analysis.md).

**Lesson.** Before optimizing memory, measure how much time is actually spent on it.

### 4. Small measurement slices, with wall-clock time

**State:** replaced by larger slices measured in worker time.

**What was expected.** The automatic width choice ([03-automatic-width-selection.md](03-automatic-width-selection.md)) runs pieces ("slices") of the first launch with each candidate and keeps the fastest. Small slices would decide earlier and waste less time on bad candidates.

**What happened.** With small slices measured by the wall clock, the choice was **wrong**: it even chose the cooperative 128-lane variant (`128c`) where it was not the best. In tiled matmul, slices of 8 blocks per worker chose a variant that was **40% slower** over the whole launch.

**Why.** In a short slice, the dominant variance does not come from the width but from transient effects: processor frequency transitions (boost), cold caches and TLB at the start of the slice, and the uneven tail (some workers finish earlier and sit idle). With the wall clock, this tail enters the computation. The solution was to measure the **time the workers spent working, divided by the number of blocks**, with 32 blocks per worker in each slice, and two rounds per candidate, keeping the best.

### 5. "Worker time" for comparing whole launches

**State:** replaced by wall-clock time, in this case only.

**What was expected.** After item 4, worker time seemed the right metric for everything. It was also used when the launch is too small to be sliced and each candidate is measured in a whole launch.

**What happened.** In `matrix-rotate`, it chose **the worst width**.

**Why.** Worker time measures the total work (sum of the busy times), not the latency of the launch (makespan). In `matrix-rotate` (only 10 blocks, see [16-splitting-blocks-into-warps.md](16-splitting-blocks-into-warps.md)), the 128-lane variant did **less work** in total, but, by grouping more threads per task, left cores idle. By worker time it won; by the clock, it lost. When whole launches are compared with each other, what matters is precisely the wall-clock time, which already includes how well the work spreads across the workers. The two metrics coexist: worker time for slices within a launch, the clock for whole launches.

### 6. Contiguous ranges per worker

**State:** replaced by interleaved chunks.

**What was expected.** Give each worker a contiguous, fixed range of the grid, so that, in kernels repeated over the same data, each core finds in its L2/L3 the data that it itself touched in the previous launch.

**What happened.** The `iso2dfd` stencil improved a lot (1.27 to 0.42 ms per step), but matrix multiplication **got worse**: naive from ~14 to ~25 Gcyc, tiled from ~15 to ~20 Gcyc.

**Why.** The matmul block order ([04-block-order.md](../iteration-1/04-block-order.md)) counts on all workers working on **neighboring** blocks **at the same time**, sharing rows and columns of the matrices in the cache. With contiguous ranges, the workers are spread over the entire grid. The solution (chunks of 64 blocks distributed round-robin) is in [17-block-distribution.md](17-block-distribution.md).

### 7. The cooperative variant as the default

**State:** kept as a **candidate**, not as the default.

**What was expected.** In the cooperative variant ([15-smt-cooperative-variant.md](15-smt-cooperative-variant.md)), the two sibling CPUs of a core (SMT) each run half of a block, at the same time, synchronizing at barriers. Since they share the L1D and the barrier between the two is cheap (a cache line local to the core), it seemed advantageous for every kernel with barriers and shared memory.

**What happened.** On several kernels it wins, but on matmul it **loses**.

**Why.** In tiled matmul, each iteration has a **load** phase (copying a piece of the matrices into shared memory) and a **compute** phase, separated by barriers. When the two SMT siblings run independent blocks, their phases become unsynchronized: one is in the load, limited by memory latency, while the other is in the compute, using the FMA units, and SMT overlaps the two things. In the cooperative variant, the barrier aligns the phases: both load **at the same time** and then compute at the same time, competing for the same units in each phase, and the memory latency is no longer hidden. For this reason it enters the contest of the automatic width choice, and the system picks it only where it wins.

### 8. OpenMP with 1 thread per team (in the comparison)

**State:** configuration reverted. It is not our code.

**What was expected.** To compare the simulator with the OpenMP version of the HeCBench programs, we tried to configure OpenMP with one thread per team (`teams distribute`, the OpenMP analog of a block), to approximate the block model.

**What happened.** This configuration **broke** the `fdtd3d`, `babelstream` and `triad` programs in the OpenMP version of HeCBench (wrong results or failures).

**Why.** These programs, in the OpenMP version, assume more than one thread per team at some point. The problem is not in the simulator, but it affected the comparison table: a wrong reference number makes any result look better or worse than it is.

### 9. Cooperative barrier as a function call

**State:** replaced by inline code.

**What was expected.** The first version of the barrier between the two siblings of the cooperative variant was a runtime **function**, called by the generated code at each `__syncthreads()`. Simple to write and to debug.

**What happened.** The generated code became full of zmm register reads and writes on the stack: **833 accesses** to zmm on the stack in one kernel.

**Why.** In the System V x86-64 ABI there are no callee-saved vector registers: all of `zmm0`-`zmm31` (and the mask registers `k0`-`k7`) are caller-saved. Before each opaque call, the generated code must save on the stack every zmm that is live across it and reload it afterwards, 64 bytes per register. In a vectorized kernel almost all live values are in zmm, so each `__syncthreads()` cost dozens of spills and reloads. The solution was to generate the barrier **inline** (inside the code itself): a few atomic accesses and a wait loop, with no call at all.

## The code

Some of these items left marks in the current code.

### The block function, off by default

`src/device/device_core.cpp:396-400`

```cpp
            lanes = pick_width(fn->second, cfg, &entry, &tuning_key, &trials);
            // Opt-in: measured slower than per-warp calls, and it does not use
            // barrier_pass (the last warp to arrive still suspends).
            static const bool use_block = std::getenv("CPUSIM_BLOCK_FN") != nullptr;
            if (lanes == CPUSIM_WARP_SIZE && use_block) block_fn = fn->second.block;
```

The block function continues to be generated, but is only used if the environment variable `CPUSIM_BLOCK_FN` exists, and only on the 32-lane path (item 1).

### The size of the measurement slices

`src/device/device_core.h:148-158`

```cpp
    // Blocks per candidate when the width is measured inside one launch: 32
    // per worker (CPUSIM_TRIAL_BLOCKS), a few ms per slice. Shorter slices
    // measured clock and cache noise more than the width (matmul_tiled: 8 per
    // worker picked a variant 40% slower over the whole launch).
    uint64_t trial_blocks() const {
        static const uint64_t per_worker = [] {
            const char* v = std::getenv("CPUSIM_TRIAL_BLOCKS");
            return v ? std::max(1, std::atoi(v)) : 32;
        }();
        return per_worker * executor_->workers();
    }
```

The record of item 4 became a comment: 32 blocks per worker, because 8 chose wrong.

### The clock for whole launches

`src/device/device_core.cpp:494-502`

```cpp
    // One candidate per launch (launches too small for in-launch trials):
    // whole launches of one shape compared with each other, so the wall time
    // is the right measure, and the one that matters: with few blocks it also
    // counts how well the width spreads over the workers (matrix-rotate: 128
    // lanes does less work in all but leaves cores idle). The launch overhead
    // is the same for every candidate. Per block, for grids that vary.
    const uint64_t launch_blocks = uint64_t(cfg.grid[0]) * cfg.grid[1] * cfg.grid[2];
    if (tuning_key && trials.empty() && launch_busy)
        record_timing(id, function, tuning_key, lanes, result->kernel_ns * 1000 / std::max<uint64_t>(launch_blocks, 1));
```

`result->kernel_ns` is the wall-clock time of the launch. This is the fix for item 5.

### The cooperative as a candidate, after the wide ones

`src/device/device_core.cpp:551-558`

```cpp
    // With barriers, the wide variants come first: the 32-lane one, which
    // suspends at every barrier, has been the slowest every time we measured
    // (matmul_tiled, scan, heat2d, fdtd3d), and a program too short for the
    // measurement to finish keeps running the first candidates.
    if (barriers) std::stable_sort(valid.begin(), valid.end(), [](uint32_t a, uint32_t b) {
        auto rank = [](uint32_t w) { return w == CPUSIM_WARP_SIZE ? 3 : (w & kSuspend) ? 2 : (w & kCoop) ? 1 : 0; };
        return rank(a) < rank(b) || (rank(a) == rank(b) && (a & 0xffff) > (b & 0xffff));
    });
```

The trial order is: wide variants (position 0), cooperative (1), wide with suspension (2), 32 lanes (3). The cooperative one takes part, but is not the first choice (item 7).

### The inline cooperative barrier

`src/compiler/simt.cpp:1816-1840`

```cpp
// Cooperative variant: wait for the other half of the block, inline.
//     e = *epoch + 1; atomic store-release *epoch = e;
//     while (atomic load-acquire *other < e) pause;
// Only when some lane of this half is here, so the two halves count the same
// barriers (a half that already returned stops counting, see kernel_abi.h).
void WarpBuilder::emit_coop_barrier(Value* mask) {
    // ...
    Value* epoch = field(CPUSIM_CTX_COOP_EPOCH, "coop.epoch");
    Value* other = field(CPUSIM_CTX_COOP_OTHER, "coop.other");
    LoadInst* cur = b_.CreateAlignedLoad(i64, epoch, Align(8), "coop.cur");
    cur->setAtomic(AtomicOrdering::Monotonic);
    Value* next = b_.CreateAdd(cur, ConstantInt::get(i64, 1), "coop.next");
    b_.CreateAlignedStore(next, epoch, Align(8))->setAtomic(AtomicOrdering::Release);
    b_.CreateBr(wait);
```

The barrier is an epoch protocol between the two siblings: each half publishes with a store-release the number of the barrier it reached and waits, with `pause`, until the load-acquire of the other half's counter reaches the same value. The release/acquire pair guarantees that the shared memory writes made by one half before the barrier are visible to the other after it. With no function call, the zmm registers stay in registers (item 9).

## When it applies (and when it does not)

Not everything here is "never do this":

- The **block function** and the **cooperative variant** still exist. They may become worthwhile again if the generated code changes (for example, with less register pressure).
- The **whole-block variant** only turned out bad **without** JIT; with JIT, it is one of the best.
- The measurement lessons (items 4 and 5) hold for any future measurement: short slices measure noise; worker time and wall-clock time answer different questions.
- Item 3 holds as a general rule: measure the fraction of cycles of a component before optimizing it.

## Results

The recorded numbers, all cited above:

| Item | Number |
|---|---|
| 3. Shared memory | actual accesses were ~7% of the cycles |
| 4. Small slices | 8 blocks per worker chose a variant 40% slower in tiled matmul |
| 6. Contiguous ranges | iso2dfd 1.27 to 0.42 ms; matmul naive ~14 to ~25 Gcyc; tiled ~15 to ~20 Gcyc |
| 9. Barrier by call | 833 accesses to zmm on the stack |

For items 1, 2, 5, 7 and 8, the record is qualitative (which one became slower, which choice was wrong); there is no table of isolated measurements here.

## See also

- [01-specialized-jit.md](../iteration-1/01-specialized-jit.md): what made the whole-block variant work (item 2).
- [02-width-variants.md](../iteration-1/02-width-variants.md) and [03-automatic-width-selection.md](03-automatic-width-selection.md): the variants and the measurement (items 2, 4, 5).
- [04-block-order.md](../iteration-1/04-block-order.md) and [17-block-distribution.md](17-block-distribution.md): why the contiguous ranges failed (item 6).
- [09-state-at-barriers.md](../iteration-1/09-state-at-barriers.md) and [10-barrier-pass.md](10-barrier-pass.md): the per-warp calls that the block function tried to replace (item 1).
- [12-static-access-analysis.md](12-static-access-analysis.md): the real problem with shared memory (item 3).
- [15-smt-cooperative-variant.md](15-smt-cooperative-variant.md): the cooperative variant and its barrier (items 7 and 9).
- [16-splitting-blocks-into-warps.md](16-splitting-blocks-into-warps.md): `matrix-rotate` and its 10 blocks (item 5).
