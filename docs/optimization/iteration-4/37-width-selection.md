# 37. The width selection, the same way every time

> **In one sentence:** the width a kernel ran at was being decided by noise, with the same width measuring 2 to 4 times apart from one launch to the next; the selection now measures each launch's critical path in CPU time, warms up first, decides on the median of five interleaved rounds, settles near ties by a fixed order, does not measure short launches at all, and keeps its measurement in the driver across programs: 62 of 73 kernels are within 10% of the best forced width (45 before), and a one-launch program reaches its best width on later runs.

The selection measures the candidate widths of a kernel and block shape in its first launches and keeps the fastest ([03](../iteration-2/03-automatic-width-selection.md)). Two requirements drove this rework: the choice must be close to the best width, and the same program must get the same choice every time, including after the driver restarts.

## Measuring how good it is

Two scripts measure the selection on the 49 tracked programs ([tutorial 6](../../tutorials/06-tracking.md)). `bench/track/oracle.sh` runs every program on a fresh driver once with the automatic selection and once per forced width (`CPUSIM_LANES`), and `oracle.py` compares, kernel by kernel, the automatic time with the best forced one. `stability.py` reads the driver's width messages from five fresh drivers and lists the kernels that got more than one width. The first oracle, with the selection of iteration 2:

| auto against the best forced width | kernels (of 73 over 1 ms) |
|---|---|
| within 5% | 39 |
| 5 to 10% | 6 |
| 10 to 25% | 9 |
| 25 to 50% | 6 |
| 1.5 to 2x | 9 |
| over 2x | 4 |

The measurement itself was the problem. In the driver's log (`CPUSIM_VERBOSE=1`, which now prints each candidate's median and the range of its samples), groupnorm's 256-lane variant measured from 69 to 319 us per 1000 blocks across runs, and adam's 64-lane one from 2672 to 8264. With samples that spread, the choice between widths 10 to 40% apart came out at random.

## Where the noise came from, and what removes it

| Cause | Change | Code |
|---|---|---|
| A small launch was timed by its wall time, which includes the socket round trip and waking the workers (tens of microseconds, variable) | The launch's **critical path**: the most time one worker spent on the launch's blocks. It still counts what an uneven spread over the workers costs (matrix-rotate: 128 lanes does less work but leaves cores idle) | `Executor::run`, `span_ns` |
| Another process takes a worker's core for a while: wall time grows, the kernel did not get slower | The worker's time is **CPU time** (`CLOCK_THREAD_CPUTIME_ID`), for the critical path, the in-launch slices and the work of a launch | `src/device/executor.cpp:36`, `:405-436` |
| The first launch of a kernel pays the first touch of its data and cold caches, and its candidate was dropped for it (adam: 32 lanes measured 2.9x slower than 128 in its first launch, and was faster) | A **warm-up launch** that is not measured | `WidthTuning::record` |
| A single sample decided: the best of two, and after one round a cut at 1.5x of the best (single launches were seen 2x off the rest of their candidate's) | **Five rounds**, every candidate once per round, interleaved, so that a drift of the clock or of the machine's load hits all of them alike; each candidate's **median**; after the first round only a candidate **3x** slower than the best is dropped (a 20x slower one, adam's 32r, would cost 20 launches per round) | `src/device/width_tuning.cpp:62-110` |
| Two widths 5% apart: whichever measured faster won, one way in one run and the other way in the next | **Ties by order**: among the candidates within 10% of the best median, the first in the order of `valid_widths` (without barriers 32, then wider, then refill; with barriers the block-wide one, then the suspending ones, then 32). A near tie is decided the same way every time; only a clear difference is decided by the times | `WidthTuning::decide` |
| A launch of a few microseconds (HeCBench is, `rank_gpu_kernel_6`: 12 us): the executor calls as many workers as the kernel's learned cost justifies, and that estimate moves from launch to launch, so the candidates did not even run with the same parallelism; it picked 128 lanes on one driver and 64 on the next | **Short launches are not measured**: when the first round's launches did less than 1 ms of work in all (the workers' CPU time summed), the first candidate in the order is chosen | `kShortNs` in `src/device/width_tuning.h:80` |
| A program with one launch, or ten, never finished measuring: each program started over and ran the first candidate | The measurement belongs to the **driver**, per kernel code (a hash of its module's content and its name), block shape and number of workers (`WidthKey`, `src/device/width_tuning.h:125`); the next program running the same code continues it. Once decided, the samples are dropped and only the width stays. Nothing is written to disk: a wrong decision cannot outlive the driver | `DeviceCore::tuning`, `src/device/device_core.cpp:743` |

The rule itself (`src/device/width_tuning.cpp:62-70`):

```c++
size_t WidthTuning::decide(const std::vector<std::vector<uint64_t>>& times) {
    std::vector<uint64_t> med(times.size(), UINT64_MAX);
    uint64_t best = UINT64_MAX;
    for (size_t i = 0; i < times.size(); ++i)
        if (!times[i].empty()) best = std::min(best, med[i] = median(times[i]));
    for (size_t i = 0; i < times.size(); ++i)
        if (med[i] != UINT64_MAX && double(med[i]) <= double(best) * (1.0 + kTie)) return i;
    return 0;
}
```

`WidthTuning` is a class of its own, with no executor and no clock: the device feeds it times. `tests/test_width.cpp` drives it with made-up times, including 1000 random cases where a 40% difference must go to the faster width and a 5% one, with every launch off by up to 2%, must never change ([39](39-organization-and-tests.md)).

## Results

The oracle with the critical path, the warm-up, the medians, the tie rule and short launches (before the measurement moved to the driver and to CPU time):

| auto against the best forced width | before | after |
|---|---|---|
| within 5% | 39 | 55 |
| 5 to 10% | 6 | 7 |
| 10 to 25% | 9 | 6 |
| 25 to 50% | 6 | 2 |
| 1.5 to 2x | 9 | 2 |
| over 2x | 4 | 1 |

Of the three kernels still over 1.5x, two are noise in the forced runs themselves: haccmk's and knn's insertion sort ran the same code under two forced widths (one a fallback to 32) and measured 220 against 406 ms and 336 against 480. The third, rsbench's `lookup`, is a program of one launch, which the measurement across programs then fixed:

| Program | Launches | Runs on one driver until decided | Kernel time after |
|---|---|---|---|
| rsbench | 1 | 12 (32 lanes, 5.9 s) | 2.3 s (32r), 2.6x faster |
| nqueen | 10 | 2 (9.3 s) | 6.0 s (32r), 1.5x faster |

## Do fresh drivers decide alike

Five fresh drivers, one run of each tracked program on each (`stability.py`), counting the kernel shapes that got more than one width:

| Selection | Shapes with more than one width |
|---|---|
| critical path, medians of 3 rounds, drop at 2x, ties within 10% | 18 of 57 |
| medians of 5 rounds, drop at 3x | 21 of 53 |
| short launches unmeasured | 16 of 48 |
| CPU time (machine loaded by other users: load average 32 to 40) | 15 of 53 |
| CPU time, machine quiet (load average 0.1) | 17 of 53 |

Measuring in CPU time and a quiet machine did not change the count: the alternation does not come from other users' work. The kernels that still alternate are mostly near ties: the widths they alternate between are within 5 to 15% of each other (the median gap 1.10), so either choice is within about 15% of the best. Wider tie margins do not remove them: simulated on the same samples, a 30% margin still leaves 11 of 48 alternating and costs up to 30%, a 40% margin 7 of 48. In a few kernels the gap between the same two widths changed from one driver to the next (hadamard: 1.00 on four drivers and 2.25 on the fifth), which no rule over that driver's samples can absorb.

## Not done

- **A choice that cannot depend on the machine's state at all** would have to come from the kernel, not from timing it: a model from the kernel's features (`kernel_features`, [35](35-compile-time.md)) that picks the width, with measurement only to veto a choice much slower than another. Pressure alone was 1.19 to 1.41x behind measuring.
- **The executor's learned worker count** (`Executor::cap_for`) moves while a kernel is being measured. Freezing it during the measurement would let short launches be measured too.

Back to the [iteration 4 index](README.md).
