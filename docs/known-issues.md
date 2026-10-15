# Known issues

Failures seen once or now and then, not reproduced on demand, kept here to look into. Each entry: what was seen, where, and what is known.

## `cudaMemset` "invalid argument" in `sync_locks`

- **Seen:** 2026-10-13, Xeon Gold 6548Y+ server, the first full stress run (`tests/stress/run.sh both`). `spinlock_block_critical` failed at `stress.cuh:136` (`Buffer::zero`, `cudaMemset(p, 0, n * sizeof(T))`), and the rest of the program did not run.
- **Not reproduced:** 15 runs right after, the second full run, and a run of the 58 programs in parallel all passed.
- **Found (2026-10-14):** not the driver: the client's own check. A client reserves the device's address range in its address space at connection (so a host pointer can never equal a device pointer); when the range was taken, it refused every device pointer in `cudaMemset` (and every call that checks `is_device_pointer`). The range was at 0x600000000000, inside the span where the kernel randomizes the load address of PIE executables (from 0x555555554000 up to 16 TiB above with `vm.mmap_rnd_bits = 32`): 8 of 400 runs of a minimal program had their own code there. The VRAM is now at 0x200000000000, below every randomized span: 0 of 400. A client that cannot reserve the range now says so. The same cause made HeCBench groupnorm fail once (`cudaMemset ... invalid argument`) in a tracking run.

## `j0` and `y0` (double) error on the Xeon server

- **Seen:** always on both machines (these are the stress suite's two failing cases), but on the Xeon the error measured 389 and 553 ulp where the laptop's was about 10 (CUDA's bound: 7).
- **Known:** both use glibc's `j0`/`y0`, which lose accuracy near the zeros of the functions; why the error is so much larger on the server (same Ubuntu release, different CPU and glibc build flags?) is not known. An implementation of our own around the zeros would fix both.

## A driver that did not stop

- **Seen:** 2026-10-13, laptop: `cpusimd --stop` did not stop a driver started 40 minutes earlier, and a program connecting to it hung; `kill -9` was needed.
- **Known:** that driver had been built with a deadlock fixed minutes later (`learn_cost` taking a lock `run()` held), so it was probably that build. If it happens with a current build, the driver's log (`<socket>.log`) and a `gdb -p` backtrace of its threads are what to collect.

## Timing-sensitive tests on machines with power management

- **Seen:** `cuda_async` failed on the Xeon with the `schedutil` governor: two kernels in two streams did not overlap enough. With the `performance` governor it passed every time.
- **Seen:** 2026-10-13, laptop (`balance_performance`): `cuda_async` failed once in `ctest -L features -j6` and once alone ("kernels of two streams overlap", 98 against 81 ms), then passed alone. Same check, same cause as above as far as known.
- **Known:** the test compares durations; a core that ramps its clock from idle makes the first kernel slower than the second. Not a simulator bug; the test could warm the cores first.

## HeCBench programs that still fail (2026-10-14 sweep of 305 programs)

A sweep compiled the CUDA version of 305 HeCBench programs (the ones that check their own results, need no data files and no CUDA libraries) with nvcc and with cucc on the Xeon server, and ran them on the GPU, on our executor with the measured width, and with `CPUSIM_LANES=32r`. What it found and is still open:

- **bh (ECL-BH), wrong pointer in `TreeBuildingKernel`.** Every size, every width: a gather reads an address that is no allocation. The kernel builds an octree with locks taken by `atomicCAS` on `volatile int* childd` (`-2` = locked) that threads of any block spin on; its correctness rests on the order in which the other threads' writes become visible. Not investigated further yet.
- **Compilation of minutes: sc, segsort, hadamard** (fixed). Each kernel compiles in a process of its own (segsort and scan: seconds); loops with a barrier stay rolled (sc: 499 s to 3 s, and it ran instead of hanging); threads with more than 64 live values get no variants of 128 lanes or more, which never won past 47 (hadamard: 1200 s to 29 s). Tracked in bench/track (rolled.csv, heavy.csv).
- **`-dc` (separate compilation of device code), fresnel.** Device functions defined in one `.cu` and called from kernels in another need the device code of all files linked before cpusim-devc runs; cucc compiles each file on its own.
- **PTX in `asm` statements, bitcracker (`lop3.b32`).** Inline PTX is not translated; about a dozen programs in HeCBench use some (`cp.async`, `ld.acquire`, `tanh.approx`, `lop3`).

Failures that are the programs' own bugs, which the GPU hides by its timing:

- **histogram (float4 test), bin 0 sometimes off.** `histogram_gmem_accum` tests `i > ACTIVE_CHANNELS * NUM_BINS` instead of `>=`: thread 768 writes one element past `d_hist`, and our allocator places the next allocation (the partial histograms) right there, so bin 0's partial can be overwritten before it is read. With `>=` it passed 10 of 10.
- **layernorm, kernel 0 with blocks of 64 and 32-lane calls.** `BlockReduce` is called twice with the same `__shared__` buffer and no barrier between the first call's `return smem[0]` and the second call's write of `smem[0]`. A 32-lane call runs warp 0 up to its next barrier before warp 1 reads the first result. The block-wide variant (the one the measurement picks) passes.

## base64e failed once in a tracking run

- **Seen:** 2026-10-14, Xeon server, `bench/track/track.sh build2 base3`: one of the three runs of HeCBench base64e failed its check (the build before shape classes).
- **Not reproduced:** 16 more runs, 8 with that build and 8 with the next, all passed. The run's log was not kept; `track.sh` should keep the log of a failing run.

## concurrentKernels crashed once in a tracking run

- **Seen:** 2026-10-14, Xeon server, a tracking run (`widesplit`, a build reverted since): one of three runs of HeCBench concurrentKernels died with a core dump and no output at all.
- **Not known yet:** the program launches kernels of 1 thread into several streams with `cudaStreamWaitEvent` between them. No output at all suggests a crash early (the connection, the first launches) rather than in a kernel, whose fault would have been reported. The log is in that run's results (`widesplit.concurrentKernels.fail2.log`, on the server).
