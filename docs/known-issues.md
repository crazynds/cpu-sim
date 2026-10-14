# Known issues

Failures seen once or now and then, not reproduced on demand, kept here to look into. Each entry: what was seen, where, and what is known.

## `cudaMemset` "invalid argument" in `sync_locks`

- **Seen:** 2026-10-13, Xeon Gold 6548Y+ server, the first full stress run (`tests/stress/run.sh both`). `spinlock_block_critical` failed at `stress.cuh:136` (`Buffer::zero`, `cudaMemset(p, 0, n * sizeof(T))`), and the rest of the program did not run.
- **Not reproduced:** 15 runs right after, the second full run, and a run of the 58 programs in parallel all passed.
- **Known:** the error comes from the driver's address check (`DeviceCore::translate`): the pointer was not found among the context's allocations. Suspects: a context lookup or allocation map read while another program (the previous one in the suite) closes its context, or a reused process id.

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
- **Compilation over 10 minutes: sc, segsort, hadamard.** cpusim-devc (or the specialized variants) on these sources did not finish within the sweep's limit.
- **`-dc` (separate compilation of device code), fresnel.** Device functions defined in one `.cu` and called from kernels in another need the device code of all files linked before cpusim-devc runs; cucc compiles each file on its own.
- **PTX in `asm` statements, bitcracker (`lop3.b32`).** Inline PTX is not translated; about a dozen programs in HeCBench use some (`cp.async`, `ld.acquire`, `tanh.approx`, `lop3`).

Failures that are the programs' own bugs, which the GPU hides by its timing:

- **histogram (float4 test), bin 0 sometimes off.** `histogram_gmem_accum` tests `i > ACTIVE_CHANNELS * NUM_BINS` instead of `>=`: thread 768 writes one element past `d_hist`, and our allocator places the next allocation (the partial histograms) right there, so bin 0's partial can be overwritten before it is read. With `>=` it passed 10 of 10.
- **layernorm, kernel 0 with blocks of 64 and 32-lane calls.** `BlockReduce` is called twice with the same `__shared__` buffer and no barrier between the first call's `return smem[0]` and the second call's write of `smem[0]`. A 32-lane call runs warp 0 up to its next barrier before warp 1 reads the first result. The block-wide variant (the one the measurement picks) passes.
