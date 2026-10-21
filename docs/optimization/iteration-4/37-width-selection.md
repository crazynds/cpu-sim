# 37. The width selection, the same way every time

> **In one sentence:** the width a kernel ran at was being decided by noise, the same width measuring 2 to 4 times apart from one launch to the next; the selection now measures each launch's critical path in CPU time, warms up first, estimates each width from the lower half of up to 15 interleaved rounds, settles anything within 25% of the best by a fixed order, stops early when the choice is clear, does not measure short launches, and keeps its measurement in the driver across programs: 65 of 73 kernels are within 10% of the best forced width (45 before), and fresh drivers that decided differently for 21 of 53 kernel shapes now do so for 3 to 6 of 51, always between widths within about 25% of each other.

The selection measures the candidate widths of a kernel and block shape in its first launches and keeps the fastest ([03](../iteration-2/03-automatic-width-selection.md)). Two requirements drove this rework: the choice must be close to the best width, and the same program must get the same choice every time, including after the driver restarts.

## Measuring how good it is

Two scripts measure the selection on the 49 tracked programs ([tutorial 6](../../tutorials/06-tracking.md)). `bench/track/oracle.sh` runs every program on fresh drivers, once with the automatic selection and once per forced width (`CPUSIM_LANES`), every mode of a program back to back so that they see the same load on the machine, and `oracle.py` compares, kernel by kernel, the automatic time with the best forced one. `stability.py` reads the width messages of five fresh drivers and lists the kernels that got more than one width. The first oracle, with the selection of iteration 2:

| auto against the best forced width | kernels (of 73 over 1 ms) |
|---|---|
| within 5% | 39 |
| 5 to 10% | 6 |
| 10 to 25% | 9 |
| 25 to 50% | 6 |
| 1.5 to 2x | 9 |
| over 2x | 4 |

The measurement itself was the problem. In the driver's log (`CPUSIM_VERBOSE=1`, which now prints each candidate's estimate and the range of its samples), groupnorm's 256-lane variant measured from 69 to 319 us per 1000 blocks across runs, and adam's 64-lane one from 2672 to 8264. With samples that spread, the choice between widths 10 to 40% apart came out at random.

## Where the noise came from, and what removes it

| Cause | Change | Code |
|---|---|---|
| A small launch was timed by its wall time, which includes the socket round trip and waking the workers (tens of microseconds, variable) | The launch's **critical path**: the most time one worker spent on the launch's blocks. It still counts what an uneven spread over the workers costs (matrix-rotate: 128 lanes does less work but leaves cores idle) | `Executor::run`, `span_ns` |
| Another process takes a worker's core for a while: wall time grows, the kernel did not get slower | The worker's time is **CPU time** (`CLOCK_THREAD_CPUTIME_ID`), for the critical path, the in-launch slices and the work of a launch | `src/device/executor.cpp:36`, `:405-436` |
| The first launch of a kernel pays the first touch of its data and cold caches, and its candidate was dropped for it (adam: 32 lanes measured 2.9x slower than 128 in its first launch, and was faster) | A **warm-up launch** that is not measured | `WidthTuning::record` |
| Few samples decided: the best of two, then a median of five | **Up to 15 rounds**, every candidate once per round, interleaved, so that a drift of the clock or of the load hits all of them alike. Each candidate's **estimate** is the mean of its times between the lowest eighth and the middle: what disturbs a launch (another process, an interrupt, a page fault) only adds time, so the noise is in the upper half; the lowest eighth goes too, against a launch that measured unusually little | `estimate`, `src/device/width_tuning.cpp:55-69` |
| A candidate dropped after one round for a single slow launch | After the first round only a candidate **3x** slower than the best is dropped (a 20x slower one, adam's 32r, would cost 20 launches per round), and **never the first** in the order, which a tie falls back to | `WidthTuning::record` |
| Two widths 5 to 20% apart: whichever measured faster won, one way in one run and the other way in the next | **Ties by order**: among the candidates within **25%** of the best estimate, the first in the order of `valid_widths` (without barriers 32, then wider, then refill; with barriers the block-wide one, then the suspending ones, then 32). A near tie is decided the same way every time; only a clear difference is decided by the times | `WidthTuning::decide` |
| 15 rounds cost a program of few launches many runs before deciding | **Early stop**: after 3 rounds, the measurement ends when the choice would be the same with every estimate 20% off either way (with these margins, a winner by about 1.9x); a closer call gets all 15 | `WidthTuning::settled` |
| A launch of a few microseconds (HeCBench is, `rank_gpu_kernel_6`: 12 us): the executor calls as many workers as the kernel's learned cost justifies, and that estimate moves from launch to launch, so the candidates did not run with the same parallelism | **Short launches are not measured**: when the first round's launches did less than 1 ms of work in all (the workers' CPU time summed), the first candidate in the order is chosen | `kShortNs`, `src/device/width_tuning.h:98` |
| A program with one launch, or ten, never finished measuring: each program started over | The measurement belongs to the **driver**, per kernel code (a hash of its module's content and its name), block shape and number of workers (`WidthKey`, `src/device/width_tuning.h:145`); the next program running the same code continues it. Once decided, the samples are dropped and only the width stays. Nothing is written to disk: a wrong decision cannot outlive the driver | `DeviceCore::tuning`, `src/device/device_core.cpp:743` |

Large launches still measure every candidate inside their first launch, on slices of 32 blocks per worker; those slices are long and steady, so they get 5 rounds (`kSliceRounds`), with the same estimate and tie rule. 15 would need launches of half a million blocks.

The estimate (`src/device/width_tuning.cpp:55-69`):

```c++
uint64_t estimate(std::vector<uint64_t> v) {
    std::sort(v.begin(), v.end());
    // (From 3 times on, at least the lowest is left out: the measurement may
    // stop after 3 rounds, and one lucky launch must not decide it.)
    const size_t lo = v.size() >= 3 ? std::max<size_t>(1, v.size() / 8) : 0;
    const size_t hi = std::max(lo + 1, (v.size() + 1) / 2);
    uint64_t sum = 0;
    for (size_t i = lo; i < hi; ++i) sum += v[i];
    return sum / (hi - lo);
}
```

`WidthTuning` is a class of its own, with no executor and no clock: the device feeds it times. `tests/test_width.cpp` drives it with made-up times ([39](39-organization-and-tests.md)): slow and lucky launches that must not decide, a clear winner stopping after 3 rounds and a near tie running 15, and 1000 random cases where a 40% difference must go to the faster width and a 5% one, every launch off by up to 2%, must always be decided alike.

## How it got there

Five fresh drivers, one run of each tracked program on each, counting the kernel shapes that got more than one width (`stability.py`; the machine is shared with other users, and its load ran from 0.1 to 45 during these runs):

| Selection | Shapes with more than one width |
|---|---|
| critical path, medians of 3 rounds, drop at 2x, ties within 10% | 18 of 57 |
| medians of 5 rounds, drop at 3x | 21 of 53 |
| short launches unmeasured | 16 of 48 |
| CPU time (load average 32 to 40) | 15 of 53 |
| CPU time, machine quiet (load average 0.1) | 17 of 53 |
| 15 rounds, mean of the middle half | 12 of 49 |
| 15 rounds, ties within 25% | 4 of 49 |
| the estimate above, the first candidate never dropped | 3 of 51 |
| early stop after 3 rounds when settled within 10% | 8 of 56 |
| early stop when settled within 20% (the selection now) | 6 of 51 |

CPU time and a quiet machine did not change the count: the alternation does not come from other users' work. More samples and the wider tie margin did. What alternates now is a few kernels whose two best widths are close to the 25% margin (cuParallelSqrt: 128 against 64 at 1.04 to 1.28 from one driver to the next; winograd: 1.11 to 1.33), and a kernel measured inside one launch, on 5 slices (initTable). For any fixed margin, a kernel whose real gap is close to it will be decided one way or the other by its measurement; a wider margin moves the boundary where fewer kernels are and makes every decision coarser.

## Results

The oracle with the selection now, every mode of a program back to back:

| auto against the best forced width | iteration 2 | now |
|---|---|---|
| within 5% | 39 | 54 |
| 5 to 10% | 6 | 11 |
| 10 to 25% | 9 | 5 |
| 25 to 50% | 6 | 1 |
| 1.5 to 2x | 9 | 1 |
| over 2x | 4 | 1 |

The two kernels still over 1.5x belong to programs of one launch (rsbench) and ten (nqueen): in the oracle's three runs per driver they do not finish measuring. Run again on one driver, they reach their best width and stay there:

| Program | Launches | Runs on one driver until decided | Kernel time after |
|---|---|---|---|
| rsbench | 1 | 12 (32 lanes, 5.9 s) | 2.3 s (32r), 2.6x faster |
| nqueen | 10 | 2 (9.3 s) | 6.0 s (32r), 1.5x faster |

(Measured before the early stop, which shortens both.)

## Not done

- **A choice that cannot depend on timing at all** would have to come from the kernel: a model from its features (`kernel_features`, [35](35-compile-time.md)) picking the width, with measurement only to correct a choice much slower than another. Pressure alone was 1.19 to 1.41x behind measuring.
- **The executor's learned worker count** (`Executor::cap_for`) moves while a kernel is being measured. Freezing it during the measurement would let short launches be measured too.
- **In-launch trials** use 5 slices; initTable, measured that way, still alternates.

Back to the [iteration 4 index](README.md).
