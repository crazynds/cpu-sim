# 17. Distributing blocks among the workers

> **In one sentence:** instead of all workers consuming blocks from a single global atomic counter, each worker receives a fixed sequence of chunks of 64 blocks, interleaved with the others' (worker w gets chunks w, w+n, w+2n...), and only steals other workers' work when its own runs out; this way the same block always lands on the same core, launch after launch, and finds its data still in the cache.

The project's general concepts are in [00-basic-concepts.md](../00-basic-concepts.md). One topology detail is central here:

- **CCD**: the Ryzen 9 8940HX has two CCDs (Core Complex Dies) of 8 cores each. Each CCD has its own 32 MB L3, shared only by the cores of that CCD; on Zen 4 the L3 is a victim cache of the CCD's L2s. There is no unified L3: a miss in the local L3 leaves the CCD through the Infinity Fabric, and even if the line is in the other CCD's L3 the transfer costs on the order of a DRAM access. In practice, the effective L3 capacity for a data set is 32 MB per CCD, and only for the cores that touched it.
- **Stencil**: a kernel in which each grid point is updated from a fixed neighborhood, repeated over several time steps on the same arrays.

## The problem

### The global atomic counter

The original design was a **global atomic counter**: each free worker does `fetch_add(1)` and executes the block whose index it received. This is dynamic self-scheduling with granularity 1: the block-to-worker mapping is not determined in advance.

The advantage is automatic balancing: slower blocks simply make the other workers consume more indices, and all finish within at most one block of each other. The contention cost (all workers fighting over the same cache line in Modified state) is small when each block lasts microseconds.

But it has a defect that only shows up when the **same kernel runs several times on the same data**.

### The iso2dfd case

`iso2dfd` (from HeCBench) is a stencil on a 2048 x 2048 grid, with 100 time steps. At each step, a kernel reads and writes about 64 MB.

The OpenMP version of the same program (running on the same CPU) took **0.58 ms per step**. Doing the arithmetic, this gives about **110 GB/s**, three times the DRAM bandwidth of this machine. Therefore, most of the traffic does not come from DRAM.

The explanation: the OpenMP loop uses `schedule(static)`. Thread 0 always processes rows 0 to 72, thread 1 rows 73 to 145, and so on, in all 100 steps. The 64 MB of the problem fit almost entirely in the sum of the two L3s (2 x 32 MB). So, in the next step, each thread finds "its" rows still in the L3 of its CCD. Almost nothing comes from RAM.

With our dynamic counter, at each launch block 1000 landed on any core, perhaps on the other CCD. The lines that were in the L3 of CCD 0 had to be fetched again by CCD 1, at DRAM cost, and the old copy in CCD 0 only served to take up space.

```text
step t:    block 1000 -> core 3  (CCD 0)  data stays in CCD 0's L3
step t+1:  block 1000 -> core 11 (CCD 1)  CCD 1's L3 does not have it: far fetch
```

Our time was **1.27 ms per step**, more than double OpenMP's.

## The idea

The goal is **cache affinity across launches**: a deterministic and stable block-to-worker mapping, so that, in the next launch of the same kernel on the same data, each worker finds in its core's L2/L3 the lines that it itself touched in the previous launch.

There is, however, a second effect, in the opposite direction: some kernels (such as matrix multiplication) gain when **all workers are on neighboring blocks at the same instant**, because neighboring blocks read the same operand lines and the first read by one worker puts in the L3 what the others of the same CCD will read right after (temporal locality between workers, not within one). This is why the "banded" block order (swizzle) described in [04-block-order.md](../iteration-1/04-block-order.md) exists.

The final solution has to preserve both.

### Attempt 1: a contiguous range per worker

The first version divided the grid into 28 contiguous ranges: worker 0 got blocks 0 to N/28, worker 1 the following ones, and so on; whoever finished its range stole work from another.

```text
grid:  [ w0 w0 w0 w0 | w1 w1 w1 w1 | w2 w2 w2 w2 | ... ]
```

For `iso2dfd` it was great: **1.27 ms to 0.42 ms** per step.

For matrix multiplication it was bad:

| Program | Before (counter) | Contiguous range |
|---|---|---|
| matmul naive | ~14 Gcyc | ~25 Gcyc |
| matmul tiled | ~15 Gcyc | ~20 Gcyc |

(Gcyc = billions of CPU cycles spent; less is better.)

The reason: the matmul swizzle arranges the blocks so that those running **at the same time** share rows and columns of the matrices. With contiguous ranges, at the same moment worker 0 is at the start of the grid and worker 27 is at the end. Nobody shares anything anymore.

### The solution: interleaved chunks

The current version cuts the grid into **chunks** of `kChunk` blocks and distributes the chunks round-robin (block-cyclic distribution, as in OpenMP's `schedule(static, kChunk)`):

```text
chunk:    0    1    2   ...  27   28   29  ...  55   56 ...
worker:   w0   w1   w2  ...  w27  w0   w1  ...  w27  w0 ...
```

Worker `w` gets the chunks `w, w+n, w+2n, ...` (n = number of workers). This gives both properties:

1. **Everyone advances through the grid together.** At the start, the 28 workers are in chunks 0 to 27, all close to one another. Then, in chunks 28 to 55. The sharing of matmul continues.
2. **Each block always goes to the same worker.** Block 1000 is in chunk 15 (with chunk size 64), which always belongs to worker 15. In the next time step, the data of that block are in the cache of the right core.

To keep the balancing of the global counter, a worker that has exhausted its sequence does **work stealing**: it consumes positions from another worker's sequence, starting with those of neighboring index. Unlike classic work stealing (Cilk/TBB deques, in which the thief takes from the opposite end to the owner), here owner and thief consume the same sequence through the same atomic counter, in the same order; there is no deque and no synchronization protocol beyond the `fetch_add`.

## How it works, step by step

Reduced example: 10 blocks, 2 workers, `kChunk = 2`.

```text
blocks:   0 1 | 2 3 | 4 5 | 6 7 | 8 9
chunk:      0     1     2     3     4
owner:     w0    w1    w0    w1    w0
```

Each worker's sequence is numbered by a position `t` (0, 1, 2...):

| Worker | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | end (`end`) |
|---|---|---|---|---|---|---|---|
| w0 | 0 | 1 | 4 | 5 | 8 | 9 | 6 |
| w1 | 2 | 3 | 6 | 7 | | | 4 |

The formula that turns (worker, position) into a block is:

```text
block = first + ((t / kChunk) * nranges + w) * kChunk + t % kChunk
```

Checking with w0, t=2: `((2/2)*2 + 0)*2 + 2%2 = (2+0)*2 + 0 = 4`. Correct.

Each worker has its own counter (`next`) that says which is the next position of its sequence. To take work:

1. The worker tries its own counter: it adds 1 atomically and, if the old value is still less than `end`, converts the position to a block and goes to work.
2. If its sequence has run out, it tries the next worker's counter, then the following one, and so on (the stealing).
3. If all have run out, there is nothing more to do.

Each `Range` occupies its own cache line (`alignas(64)`), which eliminates false sharing between counters. In normal operation each line is only written by its owner and stays in Modified state in its L1; it only migrates when there is stealing, unlike the single global counter, whose line bounces between all cores at every `fetch_add`.

## The code

### The range structure and the formula

`src/device/executor.h:107-131`

```cpp
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
```

- `next` (the first one, outside `Range`) is the simple counter left over from the old design; today only cooperative launches use it.
- `Range` is one worker's sequence: `next` says the next position, `end` the length.
- `block_at` is the formula from the example above.
- The comment records the choice of 64 and the trade-off (about 5% more cycles in matmul relative to 4).

### Taking a block (with stealing)

`src/device/executor.cpp:105-119`

```cpp
// The next block for `worker`: from its own range, then from the others'
// (starting with its neighbours, which are on the same CCD more often).
bool Executor::claim(const Job& job, int worker, uint64_t* b) {
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
    return false;
}
```

With `k = 0`, `w` is the worker itself. Only if its sequence has already run out does the loop move to `worker + 1`, `worker + 2`... Workers with nearby indices tend to be on the same CCD, so the stealing tends to stay within the same L3. The `load` before the `fetch_add` (test-and-test-and-set) avoids an atomic RMW, which would take the line in exclusive state, on an already exhausted counter; without it, each idle worker would sweep the 28 lines with writes and make them bounce between the cores.

### Is there still work?

`src/device/executor.cpp:98-103`

```cpp
bool Executor::Job::has_work() const {
    if (!ranges) return next.load(std::memory_order_relaxed) < blocks;
    for (size_t i = 0; i < nranges; ++i)
        if (ranges[i].next.load(std::memory_order_relaxed) < ranges[i].end) return true;
    return false;
}
```

Without ranges (cooperative job), it looks at the simple counter. With ranges, it is enough for one of them not to have finished.

### Filling in the ranges

`src/device/executor.cpp:360-371`

```cpp
    if (!job.cooperative) {
        // Worker w's sequence: chunks w, w + n, w + 2n... of kChunk blocks.
        job.nranges = threads_.size();
        job.ranges.reset(new Job::Range[job.nranges]);
        job.first = first;
        const uint64_t count = job.blocks - first, per_round = Job::kChunk * job.nranges;
        for (size_t w = 0; w < job.nranges; ++w) {
            const uint64_t full = count / per_round * Job::kChunk;  // whole rounds
            const uint64_t rest = count % per_round, start = w * Job::kChunk;
            job.ranges[w].end = full + (rest > start ? std::min(Job::kChunk, rest - start) : 0);
        }
    }
```

A "round" is one chunk for each worker (`per_round` blocks). Each worker receives `full` blocks from the complete rounds, plus whatever falls to it from the incomplete round at the end (`rest`): the worker whose chunk starts before the end takes up to `kChunk` blocks, the others none.

`first` allows running only a slice of the launch (used in the width measurement, see [03-automatic-width-selection.md](03-automatic-width-selection.md)). When splitting into warps is on ([16-splitting-blocks-into-warps.md](16-splitting-blocks-into-warps.md)), all of this counts units instead of blocks, with no change in the code.

### Cooperative launches

`src/device/executor.cpp:255`

```cpp
            b = job.next.fetch_add(1, std::memory_order_relaxed);
```

In the cooperative variant ([15-smt-cooperative-variant.md](15-smt-cooperative-variant.md)), a pair of sibling CPUs runs each block together, and the first of the pair takes the block with the simple counter. These jobs do not use the ranges.

## When it applies (and when it does not)

- It holds for **all non-cooperative launches**.
- The benefit of "same block, same core" only shows up when the **same kernel runs again on the same data** and those data **fit in the cache** (stencils with time steps are the typical example). In a kernel that runs only once, the distribution in chunks behaves almost like the counter: everyone advances together.
- If the blocks are very uneven, stealing takes care of the end. But stealing takes blocks from another sequence, which are not in the thief's cache; this is unavoidable.
- With grids smaller than one round (fewer than 64 x 28 = 1792 blocks), some workers have an empty sequence and start out already stealing.

## Results

### Chunk size sweep

The chunk size (`kChunk`) decides the balance between the two effects: small chunks keep the workers' execution front narrower (good for matmul); large chunks give each worker longer runs of neighboring blocks (good for stencils).

| kChunk | matmul naive (Gcyc) | matmul tiled (Gcyc) | iso2dfd (per step) | heat2d |
|---|---|---|---|---|
| 4 | 13.25 | 14.24 | 512 us | 13.2 ms |
| 16 | 13.76 | 14.25 | 493 us | noisy |
| **64** | 13.98 | 14.64 | **406 us** | tie with OpenMP |
| 256 | 15.36 | 14.42 | 473 us | |

For `heat2d`, the values between 16 and 64 were tied with OpenMP, within the measurement noise.

We chose **64**: it is the best for `iso2dfd` by a margin, and in matmul it costs about 5% more cycles than 4.

### Overall comparison

| Program | Before | After |
|---|---|---|
| iso2dfd (per step) | 1.27 ms (global counter) | ~0.41 ms (chunks of 64) |
| black-scholes | 80 ms | 63 ms |
| nbody | 885 GFLOP/s | 959 GFLOP/s |
| matmul naive / tiled | ~14 / ~15 Gcyc | 13.98 / 14.64 Gcyc |

For comparison, OpenMP took 0.58 ms per step on `iso2dfd`; with the chunk of 64 we are below that.

## Limitations and trade-offs

- **A compromise, not an optimum for each.** Matmul prefers chunks of 4; the stencil, 64. The value is a constant (`kChunk`), the same for all kernels.
- **It depends on the workers not changing.** The affinity "block 1000 always on core 15" only exists because the workers are pinned to their CPUs and the number of workers does not change between launches.
- **Concurrent kernels.** When two kernels run at the same time ([20-concurrent-kernels.md](20-concurrent-kernels.md)), a worker may be busy with the other kernel, and its blocks end up stolen. The affinity is weaker in that case.
- **Discarded:** contiguous ranges per worker (attempt 1) are recorded in [22-what-did-not-work.md](22-what-did-not-work.md).

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): cache, L3, workers.
- [03-automatic-width-selection.md](03-automatic-width-selection.md): the `[first, last)` slices of a launch.
- [04-block-order.md](../iteration-1/04-block-order.md): the swizzle that matmul takes advantage of.
- [15-smt-cooperative-variant.md](15-smt-cooperative-variant.md): the jobs that use the simple counter.
- [16-splitting-blocks-into-warps.md](16-splitting-blocks-into-warps.md): when the distributed unit is a group, not a block.
- [20-concurrent-kernels.md](20-concurrent-kernels.md): several launches active at the same time.
- [22-what-did-not-work.md](22-what-did-not-work.md): the contiguous range.
