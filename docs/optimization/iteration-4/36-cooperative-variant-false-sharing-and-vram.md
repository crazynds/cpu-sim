# 36. A variant that never won, a false sharing it hid, and a failure one run in fifty

> **In one sentence:** measured per kernel, the SMT cooperative variant never won and was 2 to 50 times slower than the best width, so it is gone; removing its fields from the executor's job put the join counter on the same cache line as the fields every block claim reads (iso2dfd 7% and groupnorm 18% slower) until the hot atomics got lines of their own; and the intermittent `cudaMemset: invalid argument` of the stress suite was the client's reservation of the device's addresses colliding with where the kernel loads PIE executables.

## The cooperative variant, removed

The cooperative variant ([15](../iteration-2/15-smt-cooperative-variant.md)) ran one block on both SMT siblings of a core, half the block on each, synchronized at barriers by an epoch counter in the warp context. It was offered to the width selection as 64c to 512c next to the block-wide and suspending variants. `bench/track/collect_widths.sh` records, for every kernel of the 47 tracked programs, the time the runtime measured for each candidate width (`CPUSIM_VERBOSE` messages in the daemon's log), and `widths.py` tabulates them. Over 52 kernel shapes the winners were 32 lanes in 21, 64 in 13, 128 in 9, 64s in 4, 256 in 3 and 128s in 2; the cooperative variant won none and was 2 to 50 times slower than the best (its sibling waits at every barrier for the other half).

It is gone from the compiler (the entries and the inline epoch barrier), the runtime's candidates, the executor (sibling pairs running one block, `run_pair`, cooperative jobs) and the ABI: the meta bit is removed and the context's two epoch fields are unused, with every other field at its old offset (`CPUSIM_KERNEL_ABI` 14). Binaries are 9% smaller and compile time is unchanged.

## False sharing in the job

`Executor::Job` describes one launch to the workers. Every worker that joins or leaves updates `state` (a count of workers inside, and a closed bit), and every block claim reads `cap`, `nodes` and the job's other read-mostly fields. With the cooperative fields removed, `state` moved onto the same 64-byte line as those fields, and every join and leave invalidated the line every claiming worker was reading. On kernels of many short launches the cost was visible: iso2dfd 7% and groupnorm 18% slower than before the removal. The counters now have lines of their own (`src/device/executor.h:176-185`):

```c++
// Written by every worker joining and leaving, read by the claims of
// every block: on a line of its own (sharing one with the job's
// read-mostly fields made iso2dfd 7% and groupnorm 18% slower, kernels
// of many short launches).
alignas(64) std::atomic<int64_t> state{0};
// At most this many workers inside at once (Executor::cap_for).
alignas(64) int64_t cap = INT64_MAX;
```

`busy_ns` and `done`, also written by every worker, have their own lines too. This layout beats the one before the removal: over the tracked programs, least of 3 runs, kernel time geomean 0.972 against the baseline (groupnorm 0.82, heat2d 0.79, bitonic-sort 0.80).

## `cudaMemset: invalid argument`, one run in fifty

The stress suite's first full run on the Xeon failed once in `spinlock_block_critical` at a `cudaMemset` of a fresh allocation; later runs passed, and HeCBench groupnorm failed the same way once in a tracking run. Repeating a minimal program (allocate, memset, free) 400 times reproduced it 8 times.

The failure was the client's own check, not the driver. Unified addressing promises that a device pointer is never a valid host pointer, so a client reserves the device's address range in its own address space when it connects (`MAP_FIXED_NOREPLACE`, `PROT_NONE`); a pointer inside the range is then a device pointer. When the range could not be reserved, the client refused every device pointer, since any of them might have been one of its own. The range was at `0x600000000000`. The kernel loads a PIE executable at a random address from `0x555555554000` up to 16 TiB above it (`vm.mmap_rnd_bits = 32`), a span that contains `0x600000000000`: in 8 of the 400 runs the program's own code was there.

The VRAM now starts at `0x200000000000`, below every randomized span; a compile-time check keeps it there (`src/device/device_core.h:33-38`):

```c++
constexpr uint64_t kPreferredVramBase = 0x200000000000ull;
// The lowest address the kernel may load a PIE executable at; up to 16 TiB of
// VRAM must end below it.
constexpr uint64_t kLowestPieBase = 0x555555554000ull;
static_assert(kPreferredVramBase + (16ull << 40) <= kLowestPieBase,
              "the VRAM range would overlap where PIE executables are loaded");
```

0 failures in 400 runs, and none in 30 runs each of groupnorm, base64e and concurrentKernels (`bench/track/repeat.sh`). A client that cannot reserve the range now says so on its first use instead of failing each call silently (`src/device/daemon_client.cpp:173-184`).

Back to the [iteration 4 index](README.md).
