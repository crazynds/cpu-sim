# 03. Automatic width selection (autotuning)

> **In one sentence:** since no group width is the best for every kernel, the runtime tries the candidates in the first launches of each (kernel, block shape), measures what each one costs and starts using the fastest; this document tells how that measurement was fixed three times until it chose correctly.

> **Since then:** the selection was rebuilt in iteration 4 to decide the same way every time (critical path in CPU time, a warm-up, medians of 5 rounds, ties decided by order, short launches unmeasured, the measurement kept across programs): [37](../iteration-4/37-width-selection.md).

The project's general concepts (launch, worker, lane, Gcyc) are in [00-basic-concepts.md](../00-basic-concepts.md). The variants (32, 64, 128, 256, `s`, `c`) are in [02-width-variants.md](../iteration-1/02-width-variants.md).

> **Iteration:** the first version of the selection (one whole launch per candidate) is from iteration 1 and is described in the [original log](../iteration-1/OPTIMIZATIONS.md). This document tells how it was redone in iteration 2 and describes the current algorithm.

## The problem

The compiler generates several versions of each kernel: 32 lanes, 64, 128, whole block, suspending, cooperative. The best one depends on interacting factors that are hard to model statically: register pressure (larger widths multiply the number of zmm registers per live variable value and cause spills), divergence fraction (larger widths execute more masked lanes), number of barriers (suspension cost in the 32 and `s` variants) and number of blocks in the grid (parallelism available to 28 workers). Measured examples:

- in the naive matmul, 128 lanes beats 32 by 35%;
- in the tiled matmul, the whole-block one (256) beats all the others;
- in fdtd3d, 256 lanes loses almost 80% of its time to spills, and the best is the cooperative 128c;
- in heat2d, almost all tie, and 128 loses.

A fixed heuristic gets one of these cases wrong. The alternative is **empirical autotuning** at run time: measure the candidates and choose. The difficulty is obtaining a measurement that is at once cheap, low in noise and representative of the real cost, as the sequence of fixes below shows.

## The idea

Choosing the best of *k* variants by measurement is a selection problem under noise with an exploration cost. Three sources of systematic error appear:

- **Warm-up effects:** the first execution pays page faults (first touch of `cudaMalloc` pages), cold caches and TLB and, on the laptop, frequency transitions (boost and thermal/power limits). The candidate measured first is penalized.
- **Noise:** interrupts, other processes, clock variation and the imbalance at the end of each measured segment. Timing noise is essentially additive and non-negative (interference can only make execution slower), so the **minimum** of several samples is a better estimator of the intrinsic cost than the mean.
- **Exploration cost:** each measurement of a bad candidate is wasted time, as in the multi-armed bandit problem.

The runtime rules correspond to this: measure each candidate more than once in interleaved order (A B C D A B C D), so that warm-up and clock changes hit all of them in a similar way; keep the minimum of each one; discard early candidates that cannot win; and choose a metric that captures the relevant cost in each regime. Each rule came from an observed error.

## How it works, step by step

The versions are described in the order in which they were implemented.

### Version 1: one whole launch per candidate

The first version was the simplest possible:

1. on the 1st launch of a (kernel, block shape), run with candidate A and measure the time per block;
2. on the 2nd, with candidate B; and so on;
3. at the end, measure the first candidate **again** (because the first launch pays for a cold cache and page faults);
4. launches that were too small (fewer than 16 blocks per worker) did not count, since they did not occupy all the workers.

Two problems appeared.

**Problem A: exploration cost.** Each candidate spends a **whole** launch. In a profile of the tiled matmul with 4 launches in total, **86% of the time** was spent running the slow candidates (the 32-lane one, for example, almost 2x slower than the 256 one). The program ended before amortizing the cost of the measurement.

**Problem B: warm-up decided the choice.** The naive matmul chose 32 lanes. The decisive measurement was contaminated by the cold state of the first execution (data outside the cache hierarchy, pages not yet mapped, causing page faults on first touch), and a single re-measurement at the end was not enough to undo the distortion. With the new measurement (described below), the naive one chooses 128 and becomes 30% faster: from **21.4 to 14.1 Gcyc**.

### Version 2: measure inside the first launch, in slices

Instead of spending one launch per candidate, version 2 measures all of them **inside the first launch**, using its own blocks:

```text
blocks of the 1st launch:  [slice A][slice B][slice C][slice D][slice A][slice B][slice C][slice D][ rest with the winner ........ ]
                           |------ round 1 ----------------| |------ round 2 ----------------|
```

- each candidate runs one **slice** of blocks (for example 8 blocks per worker);
- there are **two interleaved rounds** (A B C D A B C D): warm-up and clock drift are spread among the candidates instead of systematically penalizing the first;
- of each candidate, the **minimum** of the two slices is kept;
- the rest of the launch already runs with the winner.

This is correct because the CUDA programming model defines neither order nor concurrency among the blocks of a grid (blocks can only communicate through atomics in global memory, with no guarantee of mutual progress), and all the variants implement the same per-block semantics: executing different blocks with different variants produces the same result.

**The new problem: short slices are not representative.** With 8 blocks per worker, in the tiled matmul, the 128c slice looked **2x faster** than the 256 one. But over the whole launch the 256 took **153 ms** and the 128c took **211 ms**. The choice was wrong by ~40%.

To understand this, a sweep of the slice size was made (from 1 to 32 blocks per worker). In small slices, the variance and the biases of the measurement (clock variation, cache state inherited from the previous slice, imbalance among workers at the start and at the end) were larger than the real difference between the widths. The size became **32 blocks per worker** (configurable with `CPUSIM_TRIAL_BLOCKS`). With 28 workers, each slice has 896 blocks, a few milliseconds.

### Version 3 (current): the right measure for each case

The third round of fixes changed **what** is measured.

**1. Inside a large launch: "worker time per block".** Each worker times how long it spent executing blocks of the slice; the executor sums these times (`busy_ns`) and divides by the number of blocks. The wall-clock time of the slice (from start to end) is unsuitable here:

```text
wall clock of a slice, 4 workers:
worker 1: ████████████
worker 2: ██████████████
worker 3: ██████████
worker 4: █████████████████   <- the last block happened to be slow
          |---- wall time = the slowest ----|
```

The end of a slice is uneven (some workers finish earlier and sit idle, the tail effect) and there is a fixed cost to dispatch each job to the workers (waking them and synchronizing the completion). In short slices, these two terms are a large fraction of the wall time and do not depend on the width. The sum of each worker's **busy** time ignores the waits and measures only the cost of executing the blocks.

**2. Comparing whole launches (small grids): launch wall-clock time.** When the grid is too small for slices (see the condition in the code), each candidate still runs a whole launch, and then the measure is the launch's **wall time**, divided by the blocks. Here the work measure would be wrong: in **matrix-rotate** (a kernel launched with few blocks), 128 lanes did **less total work**, but with groups of 128 lanes there were fewer units of work to distribute and several workers sat idle. The summed busy time does not see idleness; the wall time does. With the work metric, it chose 128 lanes, the worst: **12.3 ms**, against **5.7 to 6.0 ms** for the others. Among whole launches, the fixed launch cost is the same for all the candidates, so the wall clock is fair and measures what matters (including how much the width allows to be parallelized).

**3. Every launch counts.** The old rule "launches with fewer than 16 blocks per worker do not count" made **fdtd3d** (138 blocks per launch, fewer than 5 per worker) **never** measure. It stayed stuck on the first candidate, 32 lanes, which was precisely the worst. Now any launch counts.

**4. Second round only for those that can win.** In the comparison among launches, each execution of a candidate costs a whole launch. After the first round, only the candidates within **25%** of the best return for the second round. The second sample can only reduce a candidate's minimum, and the 25% threshold is a heuristic: the bet is that a candidate that far behind hardly recovers the difference in a second sample, so it is eliminated (a simple form of successive elimination). The value was not derived from the measured noise, which reached ±30% between executions in black-scholes; a candidate near the threshold may be wrongly discarded. If only one is left, there is no second round.

**5. Order of the candidates in kernels with barriers.** If the program is short and ends before the measurement finishes, all of its launches used the first candidates in the list. So the order is: **whole block, cooperative, suspending, 32 last**. The 32-lane one, which suspends at every barrier, was the slowest in all the barrier kernels measured (tiled matmul, scan, heat2d, fdtd3d).

### Summary of the current mechanism

```text
1st launch of (kernel, block):
  candidates = valid widths (sorted, if there are barriers)
  if blocks >= 2 rounds x 32 blocks/worker x workers x number of candidates x 2:
      measure everything inside this launch, in slices (worker time per block)
      choose and run the rest with the winner
  else:
      run with the 1st candidate and record the wall time per block
following launches (if not yet chosen):
  run the next candidate in the list, record
  end of 1st round: discard those more than 25% away from the best
  end of 2nd round: keep each one's best time, choose
after the choice: always the winner
```

The "x 2" means that the measurement slices occupy at most half of the launch.

## The code

### The slice size

`src/device/device_core.h:148-159`

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
    static constexpr int kTrialRounds = 2;
```

The comment records the result of the version 2 sweep. The slice is `32 x number of workers` blocks, and there are 2 rounds.

### The candidates and the order

`src/device/device_core.cpp:532-558`

```cpp
    std::vector<uint32_t> valid{CPUSIM_WARP_SIZE};
    for (int i = 0; i < 2; ++i)
        if (fn.susp[i] && (64u << i) < threads) valid.push_back(64u << i | kSuspend);
    for (int i = 0; i < 4; ++i) {
        uint32_t w = 64u << i;
        if (!fn.wide[i]) continue;
        if (barriers) {
            if (threads <= w) {
                valid.push_back(w);
                break;  // the smallest covering width is enough
            }
        } else if (w <= threads) {
            valid.push_back(w);
        }
    }
    // Half the block on each SMT sibling: blocks of w+1 to 2w threads.
    if (barriers && executor_->pairs())
        for (int i = 0; i < 4; ++i)
            if (fn.coop[i] && threads > (64u << i) && threads <= (128u << i)) valid.push_back(64u << i | kCoop);
    // With barriers, the wide variants come first: the 32-lane one, which
    // suspends at every barrier, has been the slowest every time we measured
    // (matmul_tiled, scan, heat2d, fdtd3d), and a program too short for the
    // measurement to finish keeps running the first candidates.
    if (barriers) std::stable_sort(valid.begin(), valid.end(), [](uint32_t a, uint32_t b) {
        auto rank = [](uint32_t w) { return w == CPUSIM_WARP_SIZE ? 3 : (w & kSuspend) ? 2 : (w & kCoop) ? 1 : 0; };
        return rank(a) < rank(b) || (rank(a) == rank(b) && (a & 0xffff) > (b & 0xffff));
    });
```

- It always starts with 32.
- Suspending (`s`) ones only when the block is larger than the width (otherwise they would be equal to the whole-block one).
- Whole-block: with a barrier, only the smallest that covers the block (`break`); without a barrier, every width up to the block size.
- Cooperative ones only if the executor has SMT pairs (`executor_->pairs()`) and the block fits in two halves.
- The sort gives each kind a rank: whole block 0, cooperative 1, suspending 2, 32 lanes 3. Within the same rank, larger width first.

### Deciding between slices and whole launches

`src/device/device_core.cpp:570-594`

```cpp
    uint64_t key = (uint64_t(cfg.block[0]) << 40) ^ (uint64_t(cfg.block[1]) << 20) ^ cfg.block[2];
    Context::Tuning& t = fn.tuning[key];
    if (t.candidates.empty() && !t.chosen) {
        // The measure is the workers' time per block (Executor::run), so it
        // holds for small slices and small launches alike. Two rounds, every
        // candidate once per round, keeping each one's best: warm-up and clock
        // changes hit all of them alike.
        for (int r = 0; r < kTrialRounds; ++r) t.candidates.insert(t.candidates.end(), valid.begin(), valid.end());
        // A launch with enough blocks measures every candidate on slices of
        // its own blocks and runs the rest with the winner: the choice is made
        // in the first launch. Smaller launches measure one candidate per
        // launch (every launch counts, whatever its size).
        const uint64_t blocks = uint64_t(cfg.grid[0]) * cfg.grid[1] * cfg.grid[2];
        if (blocks >= 2 * kTrialRounds * trial_blocks() * valid.size()) {
            for (int r = 0; r < kTrialRounds; ++r)
                for (uint32_t w : valid) trials->push_back({w, entry_for(w)});
            *tuning_key = key;
            *entry = trials->front().entry;
            return trials->front().lanes;
        }
    }
    uint32_t w = t.chosen ? t.chosen : t.candidates[t.ns.size()];
    if (!t.chosen) *tuning_key = key;
    *entry = entry_for(w);
    return w;
```

- `key` identifies the block shape: the choice is stored per (kernel, shape), because the same kernel may prefer different widths for different blocks.
- The candidate list is the valid list repeated twice (the two rounds).
- If the launch has enough blocks (the slices add up to at most half of it), it builds the `trials` list in the order A B C D A B C D and returns the first one.
- Otherwise, it returns the next candidate not yet measured (`t.candidates[t.ns.size()]`) or, if there is already a choice, the winner. The comment at line 573 still speaks of "workers' time per block" for both cases; in the current code, the measure among launches is the wall time (see below).

### Running the slices in the launch

`src/device/device_core.cpp:456-479`

```cpp
            if (!trials.empty()) {
                // One slice of blocks per candidate, timed per block; then the
                // rest of the launch with the winner.
                const uint64_t slice = trial_blocks();
                std::vector<uint64_t> ns;
                for (const Trial& tr : trials) {
                    uint64_t busy = 0;
                    run(tr.entry, tr.lanes, first, first + slice, &busy);
                    ns.push_back(busy * 1000 / slice);
                    first += slice;
                }
                // Best slice of each candidate (trials are the rounds back to back).
                const size_t n = trials.size() / kTrialRounds;
                size_t best = 0;
                for (size_t i = 0; i < n; ++i) {
                    for (size_t r = 1; r < kTrialRounds; ++r) ns[i] = std::min(ns[i], ns[r * n + i]);
                    if (ns[i] < ns[best]) best = i;
                }
                choose_width(id, function, tuning_key, std::vector<Trial>(trials.begin(), trials.begin() + n),
                             std::vector<uint64_t>(ns.begin(), ns.begin() + n));
                entry = trials[best].entry;
                lanes = trials[best].lanes;
            }
            run(entry, lanes, first, UINT64_MAX, &launch_busy);
```

- `run(..., first, first + slice, &busy)` executes only the blocks from `first` to `first + slice` with the candidate and returns in `busy` the sum of the workers' busy time.
- `busy * 1000 / slice` is the worker time per 1000 blocks.
- `n` is the number of candidates; `ns[r * n + i]` is the measurement of candidate `i` in round `r`. The minimum is kept.
- `choose_width` (lines 597-613) records the choice and, with `CPUSIM_VERBOSE=1`, prints the measurements.
- The last line runs the remainder of the launch with the winner.

### Recording whole launches (wall time)

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

When there were no slices (`trials.empty()`) and the launch is part of the measurement (`tuning_key`), the launch's wall time (`kernel_ns`), divided by the blocks, is recorded. There is no minimum block count: it is enough that the launch has run (`launch_busy` nonzero). Dividing by the blocks makes it possible to compare launches of grids of different sizes.

### The 25% pruning and the final choice

`src/device/device_core.cpp:622-648`

```cpp
    Context::Tuning& t = fn->second.tuning[key];
    if (t.chosen || t.ns.size() >= t.candidates.size() || t.candidates[t.ns.size()] != lanes) return;
    t.ns.push_back(ns);
    const size_t n = t.first_round ? t.first_round : t.candidates.size() / kTrialRounds;
    if (!t.first_round && t.ns.size() == n) {
        // First round done. The second one only separates candidates close
        // to the best (within 25%); the others cannot win and are not run
        // again (each run is a whole launch with that width).
        t.first_round = n;
        const uint64_t best = *std::min_element(t.ns.begin(), t.ns.end());
        t.candidates.resize(n);
        for (size_t i = 0; i < n; ++i)
            if (t.ns[i] <= best + best / 4) t.candidates.push_back(t.candidates[i]);
        if (t.candidates.size() == n + 1) t.candidates.pop_back();  // only one left: no need
    }
    if (t.ns.size() == t.candidates.size()) {
        // Each candidate's best over its runs.
        std::vector<uint64_t> best_ns(t.ns.begin(), t.ns.begin() + n);
        for (size_t k = n; k < t.ns.size(); ++k)
            for (size_t i = 0; i < n; ++i)
                if (t.candidates[i] == t.candidates[k]) best_ns[i] = std::min(best_ns[i], t.ns[k]);
        size_t best = 0;
        for (size_t i = 0; i < n; ++i) {
            t.ns[i] = best_ns[i];
            if (t.ns[i] < t.ns[best]) best = i;
        }
        t.chosen = t.candidates[best];
```

- The first line ignores out-of-order measurements (for example from a concurrent launch that ran another width).
- When the first round completes, the list is cut to the `n` candidates and only those within `best + best/4` (25% above the best) re-enter. If only one re-entered, it is already the best and the second round is canceled.
- When all the expected measurements have arrived, each candidate keeps its best time and the smallest wins.
- The pruning applies to the comparison among launches. Inside a large launch, the two rounds of slices run in full, because each slice is cheap.

### The busy time of each worker

`src/device/executor.cpp:79-93`

```cpp
        alignas(64) uint64_t counters[CPUSIM_COUNTERS] = {};
        frames.resize(size_t(job->frame_stride) * job->warps->count);
        const auto t0 = std::chrono::steady_clock::now();
        if (!job->cooperative) run_blocks(*job, index, ctx, shared, counters, frames.data());
        else run_pair(*job, ctx, shared, counters, *pairs_[pair_of_[index]], half_of_[index], round);
        const uint64_t ns = uint64_t(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        {
            std::lock_guard lock(mutex_);
            job->busy_ns += ns;
            // ...
```

Each worker, on entering a job (`job`), records the time (`t0`), executes the blocks it manages to take (`run_blocks`, or `run_pair` for the cooperative ones) and, on leaving, adds the duration to `job->busy_ns`. Waits for work do not count, because they happen outside the timed interval. The comment in `src/device/executor.h:78-81` summarizes the choice:

```cpp
    // busy_ns: the time workers spent running this launch's blocks, summed
    // over workers. Divided by the blocks, the cost of a block: unlike wall
    // time, it ignores the uneven end of a launch and the launch overhead,
    // and other launches running at the same time (streams).
```

### Diagnostic messages

`src/device/device_core.cpp:484-493`

```cpp
    // CPUSIM_VERBOSE=2: how busy the workers were (sum of their time on this
    // launch over workers x wall time), to tell load imbalance from slow code.
    static const bool verbose_launches = [] {
        const char* v = std::getenv("CPUSIM_VERBOSE");
        return v && std::atoi(v) >= 2;
    }();
    if (verbose_launches && result->kernel_ns)
        std::fprintf(stderr, "cpusim: launch %u%s lanes, %.3f ms, workers busy %.0f%%\n", lanes & 0xffff,
                     width_suffix(lanes), result->kernel_ns / 1e6,
                     100.0 * double(launch_busy) / (double(result->kernel_ns) * double(executor_->workers())));
```

The occupancy is `summed busy time / (wall time x workers)`. 100% means that all the workers executed blocks during the whole launch; a low value indicates imbalance or lack of parallelism (as in matrix-rotate with 128 lanes), not slow code.

## When it applies (and when it does not)

- **It applies** to every normal launch with more than one valid width. Launches with profiling or in debug mode do not go through here.
- **A single candidate** (for example a 32-thread block without a barrier): runs 32 lanes directly, without measuring.
- **`CPUSIM_LANES=<n>`** forces a width (`CPUSIM_LANES=128`, `CPUSIM_LANES=64s`, `CPUSIM_LANES=128c`). If the requested width is not valid for that launch, it uses 32.
- **`CPUSIM_VERBOSE=1`** shows the measurements and the choice; **`CPUSIM_VERBOSE=2`** also shows, per launch, the width, the time and the workers' occupancy.
- **`CPUSIM_TRIAL_BLOCKS=<n>`** changes the slice size (blocks per worker).
- **Very short programs** may end before the measurement finishes. This is why the order of the candidates matters: the ones that usually win come first.
- **The choice holds for the process:** it is not stored on disk; each execution of the program measures again.

## Results

Validation: for each kernel, the best width forced with `CPUSIM_LANES` compared with what the automatic selection chose in 3 executions.

| Kernel | Best forced | Automatic choice (3 executions) |
|---|---|---|
| matmul naive | 128 (13.96 Gcyc) | 128, 128, 128 |
| matmul tiled | 256 (14.35 Gcyc) | 256, 256, 256 |
| iso2dfd | 128 (384 us) | 128, 128, 128 |
| heat2d | all ~4.40 to 4.47 ms, except 128 | 32, 32, 32 |
| fdtd3d | 128c (2.57 ms) | 128c, 128c, 256 (the 2nd best, 2.88 ms) |
| matrix-rotate | 32 (~5.7 to 6.0 ms) | 32, 32, 32 (6.1 to 6.4 ms, with the measurement cost spread over 10 launches) |
| black-scholes | tied widths | choices within the noise |

Comments:

- In the clear cases (matmul, iso2dfd), the choice is always right.
- In heat2d, where almost all tie, the choice lands on one of the tied ones and avoids 128, which is worse.
- In fdtd3d, one of the three executions chose 256, the second best (2.88 ms against 2.57 ms). It is the most visible remaining error.
- In matrix-rotate, the choice is the right one; the 6.1 to 6.4 ms include the cost of trying worse candidates in some of the 10 launches.
- In black-scholes, the kernel itself varies ±30% between executions **with the same width** (50 against 67 ms). No measurement can separate widths that differ by less than that, and the choices fall within the noise.
- Most striking historical effect: the naive matmul, which with version 1 chose 32 lanes, started choosing 128 and became 30% faster (21.4 to 14.1 Gcyc).

## Limitations and trade-offs

- **The measurement costs:** slices of bad candidates occupy up to half of the first large launch; in small grids, whole launches run with worse candidates (in matrix-rotate, from 5.7 to 6.0 ms with the width forced to 6.1 to 6.4 ms with the automatic choice).
- **Noise:** the laptop's clock varies with temperature and power limits. Two rounds with a minimum attenuate it, but do not eliminate it (fdtd3d chose the 2nd best once in three).
- **Nothing is remembered between executions:** each execution of the program measures again. Storing the choice on disk, as the JIT does with the code, would be possible, but was not done.
- **Kernels whose cost changes between launches** (for example, data that changes pattern) may have the choice made in an unrepresentative launch.
- **The "worker time" measure does not see lack of parallelism**, and the "wall time" measure is sensitive to the fixed cost and to the tail of each slice. This is why each one is used only in the regime where it is not biased.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): workers, cycles and why the time varies.
- [02-width-variants.md](../iteration-1/02-width-variants.md): the candidates.
- [01-specialized-jit.md](../iteration-1/01-specialized-jit.md): the JIT reduces the candidates to those applicable to the block.
- [15-smt-cooperative-variant.md](15-smt-cooperative-variant.md): the `c` candidates.
- [16-splitting-blocks-into-warps.md](16-splitting-blocks-into-warps.md): how small grids are spread over the workers (the matrix-rotate case).
- [17-block-distribution.md](17-block-distribution.md): how the workers take blocks, including in the slices.
- [22-what-did-not-work.md](22-what-did-not-work.md): other attempts.
