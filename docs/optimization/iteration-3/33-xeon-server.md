# 33. A two-socket Xeon: the simulator against an H100 and OpenMP, and what a large machine broke

> **In one sentence:** on a 2 x 32-core Xeon Gold 6548Y+ (Emerald Rapids) with two H100 NVL, the simulator (62 cores, 124 workers) passes 631 of 633 stress cases and is faster than the same computation in OpenMP with 128 threads on 444 of 541 (geometric mean 1.98x), and about 25x slower than one H100; the machine exposed launches that cost 816 us instead of 3 (a monitor scanning 256 GiB of VRAM every second, every idle worker woken for every launch, every worker joining every job, and deep C-states), now 3.7 us.

The general concepts are in [00-basic-concepts.md](../00-basic-concepts.md); launch latency on the laptop is in [30](30-launches-and-hot-spots.md).

## The machine

| | |
|---|---|
| CPU | 2 x Intel Xeon Gold 6548Y+ (Emerald Rapids), 32 cores each, SMT: 128 logical CPUs, 2 NUMA nodes, 2 x 60 MiB L3 |
| Vector | AVX-512 with two 512-bit FMA units per core; AVX512-FP16, AVX512-BF16, VNNI; AMX (BF16, INT8); WAITPKG |
| Memory | 1 TB |
| GPU | 2 x NVIDIA H100 NVL (CUDA 12.0 toolkit from Ubuntu; driver 595) |
| Simulator | 62 cores reserved (CPUs 2-63, 66-127), 256 GiB of VRAM, memory interleaved over both nodes (`numactl --interleave=all` on the driver: transpose 1.6x faster than node-local), `performance` governor |

## Correctness

The stress suite passes 631 of 633 cases, as on the laptop (`j0`, `y0`: glibc). Two programs did not build with CUDA 12.0's `nvcc` (`__half_as_ushort` and friends are device-only before 12.2) and are now portable; one case compared an FMA-contracted device result with an uncontracted host one (`saturatef`), now computed the same way on both sides. On the H100, CUDA 12.0's math library exceeds NVIDIA's documented bounds on `cbrtf`, `sinpif`, `cospif` (1.1 to 1.4 ulp against 1) and `tgammaf` (7.8 against 5); the simulator passes them. All six `bench/` kernels produce the GPU's result bit for bit.

## Performance

Stress suite, each case's best of two runs (the launch fixes below came after this run: launch-bound cases are faster now).

The simulator against the same computation in OpenMP (all 128 threads) on the same machine:

| | All cases with a reference (541) | Reference of 1 ms or more (137) |
|---|---|---|
| Simulator faster | 444 | 107 |
| Simulator slower | 97 | 30 |
| Geometric mean (OpenMP time / simulator time) | 1.98 | 2.22 |
| Median | 2.04 | 2.20 |

| Simulator against OpenMP | All | >= 1 ms |
|---|---|---|
| 50x or more faster | 3 | 3 |
| 20 to 50x faster | 2 | 2 |
| 10 to 20x faster | 4 | 3 |
| 5 to 10x faster | 67 | 24 |
| 4.5 to 5x faster | 29 | 4 |
| 4 to 4.5x faster | 33 | 6 |
| 3.5 to 4x faster | 28 | 5 |
| 3 to 3.5x faster | 24 | 4 |
| 2.5 to 3x faster | 39 | 8 |
| 2 to 2.5x faster | 48 | 17 |
| 1.5 to 2x faster | 63 | 16 |
| 1 to 1.5x faster | 104 | 15 |
| 1 to 1.5x slower | 55 | 15 |
| 1.5 to 2x slower | 12 | 6 |
| 2 to 2.5x slower | 8 | 5 |
| 2.5 to 5x slower | 3 | 0 |
| 5 to 10x slower | 9 | 1 |
| 10 to 20x slower | 6 | 1 |
| 20x or more slower | 4 | 2 |

Against one H100 (617 cases): the simulator is faster on 18 (launch-bound or tiny cases), and the geometric mean of GPU time / simulator time is 0.043 (median 0.039), about 25x slower; 380 cases are 20x or more slower.

`bench/` kernels (ms):

| Kernel | H100 | Simulator | C++ OpenMP, 128 threads | C++, 1 thread |
|---|---|---|---|---|
| matmul naive | 27.8 | 53.8 | 280.9 | 10823 |
| matmul tiled | 19.4 | 51.3 | 120.6 (blocked) | 5095 |
| mandelbrot | 28.0 | 161.4 | 1501.8 | 143228 |
| transpose naive | 7.5 | 19.7 | 104.5 | 2376 |
| transpose tiled | 1.1 | 17.9 | 42.3 | 425 |
| heat2d step | 0.27 | 3.5 | 2.9 | 45.4 |
| scan blocks | 5.4 | 53.4 | 28.9 | 332 |
| bitonic sort | 5.5 | 86.8 | 109.5 | 7241 |

GEMM fp32 (all cores): peak 12.4 TFLOPS (`bench/peak.cu`); `samples/cpu_gemm` (warp 32) 7.9 TFLOPS; `bench/gemm16.cu` (warps of 16, micro-tile 8 x 48) 8.0 to 8.5 TFLOPS. On one core: 260 GFLOPS peak, warp 16 234 (90%), warp 32 196 (75%): with two FMA units per core, the micro-tile that warps of 16 allow (more columns per thread, fewer loads per FMA) pays off here, where on Zen 4 it gave 2%. The same CUDA source on the H100: 2.2 TFLOPS in fp32 (not written for a GPU), 82.6 TFLOPS in bf16.

## What a large machine broke, and the fixes

`lat.cu` (round's notes), microseconds per launch through the driver:

| Kernel | Before | After | H100 |
|---|---|---|---|
| empty, asynchronous | 7.2 | 2.5 to 3.7 | 2.1 |
| 8 blocks of 128 | 816 | 3.7 to 5.6 | 2.4 |
| 1024 blocks of 256 | 700 | 105 | 3.0 |

1. **The monitor scanned the whole VRAM.** `cpu-smi`'s resident memory came from `mincore` over the whole pool every second: 64M page entries for 256 GiB, most of a second on an unpinned thread that landed on a worker's CPU and held the process's `mmap_lock`. It now looks only at the allocations (freed memory goes back to the OS).
2. **Every idle worker woken for every launch.** When fewer workers were spinning than a launch wanted, it called `notify_all`: 124 threads woke on one futex for an 8-block kernel, ~150 context switches per launch, serialized on the futex's lock (41% of the time in `native_queued_spin_lock_slowpath`). Now a launch wakes at most two sleepers, and each one woken passes the wake-up on to two more while workers are missing: a tree, log2 deep.
3. **Every worker joined every job.** Joining and leaving costs each worker a few atomic operations on lines every other worker touches, serialized, across sockets; per-launch overhead grew linearly with the workers (24 us with 16 cores, 168 with 62). The executor now learns each kernel's cost per block from its launches (`busy_ns` / blocks) and calls about one worker per 2 us of expected work (`CPUSIM_TARGET_US`; unknown kernels get every worker). A job has a cap on the workers inside, and its blocks are dealt over that many sequences.
4. **Deep C-states.** An idle Xeon core enters C6, with 290 us to wake; a launch that waits for sleeping workers then takes longer than the workers' spin (200 us), and they fall asleep again before the next one. Disabling C6 (`/sys/devices/system/cpu/cpu*/cpuidle/state3/disable`) took the 8-block kernel from 816 to 71 us before the code fixes. The measurements after the fixes had it disabled; with it enabled, the 8-block kernel takes 18 to 50 us.
5. **Parking on Intel.** Idle workers parked with `MONITORX`/`MWAITX` on AMD and spun on `pause` elsewhere; on CPUs with WAITPKG they now park with `UMONITOR`/`UMWAIT` in C0.1 (`common/spin.h`, `CPUSIM_PARK=0` to turn it off).

Launch-bound stress cases after the fixes: `smith_waterman_diag_launches` (2047 launches) 1388 to 36.8 ms, `lcs_tiled_wavefront` 208 to 19.5, `floyd_warshall_blocked_shared` 28.5 to 11.0.

## To look at next

- **Launches of large grids:** 105 us for 1024 blocks of 256 threads with all 124 workers (23 us on the laptop's 28): the join protocol (one shared state word, one shared demand counter) should be per NUMA node, and the workers called could prefer one socket for small jobs.
- **C6:** the driver could hold the reserved CPUs out of deep C-states while it has work (`/dev/cpu_dma_latency` or per-CPU PM QoS), instead of relying on the machine's configuration.
- **AMX:** bf16 tensor cores reach 5.4 TFLOPS with `vdpbf16ps` here; AMX (`tdpbf16ps`, a 16 x 16 x 32 tile per instruction) is an order of magnitude more per core. It needs `arch_prctl(ARCH_REQ_XCOMP_PERM)` and tile configuration in the workers, and a fragment layout for tiles.
- **AVX512-FP16:** `__half` arithmetic is done in `float` and rounded once; native `vaddph`/`vmulph`/`vfmadd*ph` give the same bits with half the registers and twice the lanes per instruction. A compiler peephole (fptrunc of an fp op on fpext operands) when the target has FP16.
- **GEMM at scale:** one core reaches 90% of its peak, all cores 65%: two SMT workers per core competing for L1, and blocks of B shared across sockets. Measure with one worker per core and with socket-local B panels.
- **Gathers and memory-bound kernels:** transpose tiled is 16x slower than the H100 and heat2d 13x, both bandwidth-bound; scan and bitonic sort are 10 to 16x slower, shared-memory gathers and scatters.
- **Byte and atomic cases** from [30](30-launches-and-hot-spots.md) are still the slowest against OpenMP (`histogram_global_atomics`, `sobel_u8`, `spmv_csr_scalar`).
- **A one-off `cudaMemset` "invalid argument"** in `sync_locks` (1 of 17 runs, never reproduced): look at context and allocation lookups under concurrent programs.

Previous: [32](32-warp-size.md). Back to the [iteration 3 index](README.md).
