# 30. Kernel launches in microseconds, and the stress suite's slow cases

> **In one sentence:** an empty kernel took 42 us in-process and 59 us through the driver (2 us on the GPU); the time went to waking 28 sleeping workers through a condition variable and to their queueing on one mutex, and it is now 1.7 us in-process and 5 to 7 us through the driver, which made programs of many short launches up to 5x faster; atomics that hit the same addresses are now merged per warp (a global-atomic histogram 3x faster).

The slow cases come from the stress suite ([29](29-stress-suite.md)). How the measurements were made matters as much as the changes, so it comes first.

## Measuring on a laptop

The same program, run twice minutes apart, varied by up to 2x in this round. Two causes, both found the hard way:

- **Clock and temperature.** Every comparison below is paired: the previous build (`037e236`) and the new one, each with its own driver, alternated in the same session (`ab.sh` in the round's notes), three rounds, means compared.
- **OpenMP's idle threads.** The suite times an OpenMP reference right before each kernel. By default libgomp's threads keep spinning for milliseconds after a parallel region, on every CPU, the device's included; a kernel timed in that window shares its workers' CPUs with them, and the faster launches became, the more of a launch fell inside the window. The suite now sets `OMP_WAIT_POLICY=passive` (`tests/stress/run.sh`). A program that mixes OpenMP and CUDA on `cucc` should do the same.

## Launch latency

`lat.cu`, in the round's notes: an empty kernel and small ones, launched in a loop. Microseconds per launch:

| | Before, in-process | Before, driver | After, in-process | After, driver | GPU |
|---|---|---|---|---|---|
| Empty kernel, asynchronous | 42.5 | 59.0 | 1.7 to 2.2 | 5.2 to 7.4 | 2.1 |
| Empty kernel, then synchronize | 52.0 | 75.0 | 2.6 to 3.0 | 9.6 to 12.1 | 5.3 |
| 8 blocks of 128 | 49.5 | 57.7 | 5.8 to 6.7 | 11.6 to 14.6 | 2.1 |
| 1024 blocks of 256 | 90.7 | 92.4 | 19 to 21 | 23 to 27 | 4.1 |

What `perf` showed, step by step, and the change for each:

1. **System time.** About 50 context switches per launch: every launch woke the 28 workers through `start_cv_.notify_all()`, they found little or nothing to do and slept again. Idle workers now spin for up to `CPUSIM_SPIN_US` (200 us) before sleeping; where the CPU has `MONITORX`/`MWAITX` (Zen), they park on the cache line they wait for instead of looping on `pause`, which leaves the core to the SMT sibling still running blocks (`src/common/spin.h`).
2. **The mutex.** With the workers awake, they all queued on the executor's mutex to look for work (`__lll_lock_wait` at the top of the profile). A launch now posts a *demand*: `min(blocks, workers)` slots in an atomic counter. An idle worker takes a slot with a compare-and-swap before it touches the lock, so a one-block kernel wakes one worker, and sleeping workers are notified only when the spinning ones are too few.
3. **Joining a job.** Even then, the last of 28 workers joined 21 us after the launch. Joining scanned every worker's block range (`has_work()`: 28 cache lines written by other cores) under the lock. Now a job publishes itself in an atomic (`hot_`), a worker with a slot joins with one compare-and-swap on a word holding the count of workers inside and a "closed" bit, and the last one out with no block left closes it; a flag set by the first worker that finds no block replaces the scan. The lock remains for cooperative jobs, concurrent jobs from several streams, and workers that slept. (For a while it was a spin lock; when its holder was preempted by an unpinned thread landing on its CPU, every worker waiting for it stalled, and small kernels launched after an idle period took 0.28 ms instead of 0.03. It is a plain mutex again: it is off the hot path.)
4. **Small grids.** Blocks were dealt in chunks of 64: 1024 blocks were 16 chunks for 28 workers, and a dozen stole block by block from the rest. A grid too small for a chunk of 64 per worker now gets one chunk per worker.
5. **The program's side.** Its stream thread, its synchronizations and both ends of the driver's socket slept in `cv.wait`/`recvmsg` and were woken by the kernel. They now spin too, but only for `CPUSIM_WAIT_SPIN_US` (20 us): unlike the workers they are not pinned, and a thread spinning on a worker's CPU takes half of it (the scheduler shares the CPU, `sched_yield` or not). With 200 us here, kernels after a short wait ran 10 to 40% slower.

Two bugs the change exposed. A worker that left a job through a fault or a cancellation did not mark it as having no blocks left, and when it was the last one out the job never closed: the driver waited for it forever and every later program timed out (the suite's second full run found it). And the job lives on `run()`'s stack, and a worker that read `hot_` just as `run()` cleared it and checked the reader count could write into a stack frame already gone ("stack smashing detected", once in a few dozen runs). Both sides store and then load the other's variable, the pattern x86's store buffer reorders; the accesses are now sequentially consistent.

## Atomics on the same addresses

An atomic whose address differs per lane was one locked instruction per lane. When lanes share addresses (a histogram's hot bins), they now make one atomic per distinct address among the active lanes: the first remaining lane's address, the mask of lanes with the same, a reduction of their operands, one atomic, and each lane's returned value as the old value combined with an exclusive scan inside its group (`emit_atomic`, `src/compiler/simt.cpp`). CUDA leaves the order of a warp's atomics unspecified, so this is an order a GPU could pick.

## Bytes

A gather of bytes (or of aligned 16-bit values) reads the aligned 32-bit words holding them with `vpgatherqd` and shifts them out; LLVM turned it into one load per lane. The word never crosses a page, so it is readable whenever the byte is. The suite's byte cases barely moved: what limits them is something else.

- `rle_decode_bsearch` searches a 16 MiB table for every output byte (22 dependent gathers); its OpenMP reference is the plain sequential decode, a different algorithm 22x cheaper. Against the GPU running the same kernel the simulator is 6.7x slower, the usual cost of random gathers.
- `sobel_u8` and the others compute one byte per thread, promoted to 32 bits as C++ does, so a vector holds 16 of them; OpenMP's vectorized loop handles 64 bytes per instruction.

## Results

Paired runs against the previous build (mean of three, simulator time):

| Case | Before (ms) | After (ms) | Speedup |
|---|---|---|---|
| `smith_waterman_diag_launches` (2047 launches) | 120.1 | 22.2 | 5.4x |
| `lcs_tiled_wavefront` (255 launches) | 35.0 | 12.6 | 2.8x |
| `floyd_warshall_k_launches` (1024 launches) | 122.4 | 70.8 | 1.7x |
| `odd_even_merge_sort_1M` | 23.4 | 14.1 | 1.7x |
| `bitonic_sort_dups_512K` (passive OpenMP) | 9.3 | 5.4 | 1.7x |
| `jacobi2d_20iters` (passive OpenMP) | 8.1 | 5.1 | 1.6x |
| `histogram_global_atomics` | 148.7 | 50.3 | 3.0x |
| `and_or_xor` (atomics) | 12.4 | 4.6 | 2.7x |
| `unsigned_hist64` | 13.3 | 6.4 | 2.1x |

### Every case, against OpenMP

Every case of the suite with an OpenMP reference (541 of 631), the previous build (`037e236`) against this one, each run twice through the driver with passive OpenMP waiting, the two runs averaged; ratio = OpenMP time / simulator time, in bins of 0.5x up to 5x and wider beyond. Both builds pass 629 cases (`j0` and `y0` are the two left).

| The simulator, against OpenMP on the same CPU | Before | After |
|---|---|---|
| 20 to 50x faster | 6 | 11 |
| 10 to 20x faster | 7 | 6 |
| 5 to 10x faster | 18 | 34 |
| 4.5 to 5x faster | 1 | 8 |
| 4 to 4.5x faster | 5 | 10 |
| 3.5 to 4x faster | 8 | 18 |
| 3 to 3.5x faster | 10 | 17 |
| 2.5 to 3x faster | 14 | 30 |
| 2 to 2.5x faster | 26 | 45 |
| 1.5 to 2x faster | 55 | 56 |
| 1 to 1.5x faster | 83 | 100 |
| 1 to 1.5x slower | 98 | 79 |
| 1.5 to 2x slower | 81 | 33 |
| 2 to 2.5x slower | 40 | 19 |
| 2.5 to 3x slower | 19 | 11 |
| 3 to 3.5x slower | 16 | 16 |
| 3.5 to 4x slower | 14 | 9 |
| 4 to 4.5x slower | 3 | 3 |
| 4.5 to 5x slower | 2 | 5 |
| 5 to 10x slower | 17 | 14 |
| 10 to 20x slower | 9 | 7 |
| 20 to 50x slower | 2 | 6 |
| 50x or more slower | 7 | 4 |
| **Faster in total** | **233** | **335** |
| **Slower in total** | **308** | **206** |

Geometric mean of the ratio: 0.87 before, 1.24 after (median 0.89, 1.22): the simulator went from slower than OpenMP on most cases to faster.

Only the 114 cases whose OpenMP version takes 1 ms or more (below that, a launch's fixed cost weighs as much as the work):

| The simulator, against OpenMP on the same CPU | Before | After |
|---|---|---|
| 20 to 50x faster | 6 | 11 |
| 10 to 20x faster | 7 | 4 |
| 5 to 10x faster | 14 | 12 |
| 4.5 to 5x faster | 1 | 3 |
| 4 to 4.5x faster | 3 | 0 |
| 3.5 to 4x faster | 3 | 2 |
| 3 to 3.5x faster | 3 | 8 |
| 2.5 to 3x faster | 7 | 5 |
| 2 to 2.5x faster | 4 | 5 |
| 1.5 to 2x faster | 13 | 10 |
| 1 to 1.5x faster | 15 | 22 |
| 1 to 1.5x slower | 14 | 11 |
| 1.5 to 2x slower | 5 | 4 |
| 2 to 2.5x slower | 3 | 4 |
| 2.5 to 3x slower | 3 | 4 |
| 3 to 3.5x slower | 5 | 1 |
| 3.5 to 4x slower | 1 | 1 |
| 4 to 4.5x slower | 0 | 1 |
| 4.5 to 5x slower | 1 | 1 |
| 5 to 10x slower | 3 | 2 |
| 10 to 20x slower | 2 | 1 |
| 20 to 50x slower | 0 | 1 |
| 50x or more slower | 1 | 0 |
| **Faster in total** | **76** | **82** |
| **Slower in total** | **38** | **31** |

Geometric mean 1.76 before, 2.06 after (median 1.72, 1.76). The cases that changed most for the worse are math and conversion kernels of about 0.1 ms (`longlong_as_double`, `ll2float_rn`: 0.4 to 0.5x), where a single launch's timing is mostly noise.


## What is left

- **Atomics contended across cores:** `hist256_global_atomics` (256 random bins, few repeats inside a warp) improved only 1.2x: every core writes the same 16 cache lines. Merging the updates of different warps would change when they become visible to the rest of the kernel.
- **Kernels written for a GPU's memory system:** `matmul_regblock4x4` (2.8x slower than OpenMP), `spmv_csr_scalar` (2.7x), `gauss_separable_r8` (2.0x).
- **Divergence:** a per-thread quicksort (`array_by_pointer_quicksort`, 5x) and loops with per-thread trip counts.
- **Through the driver** a launch still costs 5 to 7 us, against 1.7 us in-process: a socket round trip per launch. Sending launches without waiting for each reply would remove most of it.

Previous: [29](29-stress-suite.md). Next: [31](31-gathers-and-scatters.md). Back to the [iteration 3 index](README.md).
