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
