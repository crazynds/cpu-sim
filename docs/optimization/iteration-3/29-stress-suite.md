# 29. The stress suite: what broke, what was fixed, what is slow

> **In one sentence:** 58 programs (631 cases) written to pass on a real GPU, then run through `cucc` on the driver: before this round 174 cases passed and 444 did not even compile; after it, 629 pass, the two left are a glibc accuracy limit, and the cases that run slower than OpenMP on the same CPU are listed at the end for a later round.

The suite and how to run it: [tests/stress/README.md](../../../tests/stress/README.md). The general concepts are in [00-basic-concepts.md](../00-basic-concepts.md).

## The suite

One feature area per program, so that something the simulator cannot compile hides only its own cases. Each case checks its result against a host computation and reports the device time and the time of the same computation with OpenMP on every CPU thread. The GPU is the oracle: every program passes when built with `nvcc` and run on an RTX 5060 Laptop, so a failure on the simulator is the simulator's.

| Group | Programs | Cases | What they exercise |
|---|---|---|---|
| `geo_*` | 3 | 67 | block and grid shapes (1 to 1024 threads, 2D, 3D, 2^20 blocks), parameters (structs by value, 20+ arguments, templates, functors) |
| `type_*` | 6 | 43 | 8 to 64-bit integers and their overflow, `half`, `bfloat16`, vector types, unaligned structs, NaN, -0, denormals, division by zero |
| `math_*` | 7 | 244 | every math function and intrinsic, against the ulp bounds NVIDIA documents, special values included |
| `ctl_*` | 5 | 26 | divergent control flow, `switch`, `goto`, per-thread trip counts, recursion, per-thread arrays up to 4 KiB |
| `mem_*`, `api_*` | 15 | 84 | `__device__` and `__constant__` variables, dynamic shared memory, 2D/3D copies, pinned, mapped and managed memory, streams, graphs, device `malloc`, `printf`, `assert`, queries, error codes |
| `sync_*` | 10 | 83 | atomics of every kind, locks, `__threadfence`, `__syncthreads_count`, volatile warp-synchronous code, warp intrinsics, cooperative groups, libcu++ atomics |
| `wl_*` | 12 | 83 | classic workloads: reductions, scans, transposes, stencils, linear algebra, sorts, graphs, physics, ML layers, byte processing, dynamic programming |

## Before and after

| | Cases | PASS | FAIL | COMPILE | Driver died |
|---|---|---|---|---|---|
| Before this round (commit `d96775f`) | 632 | 174 | 13 | 444 | 1 |
| After | 631 | 629 | 2 | 0 | 0 |

(One case fewer after: a program's cases are counted from its GPU run when the simulator stops before them.)

## What broke, and the fix

### Crashes and hangs

| Symptom | Cause | Fix |
|---|---|---|
| A kernel's invalid access killed the driver, and with it every program it served | a SIGSEGV in a worker thread is fatal to the process | a fault handler on each worker's alternate stack leaves the kernel with `siglongjmp`; the launch fails with `cudaErrorIllegalAddress` (a trap with `cudaErrorAssert`), sticky in that program's context as on a GPU (`src/device/executor.cpp`) |
| One hung program hung the driver for everyone | a kernel that never ends kept the workers after its program was killed | the driver checks each second for programs that are gone and cancels their kernels (a signal to the workers inside kernel code); closing a context waits for its kernels first |
| Spin locks and flag waits hung | the waiting lanes and the lane that would write run in the same vector (or the waiting branch was emitted before the writing one), so the writer never runs | loops whose exit depends on a volatile or atomic read get a yield point (the warp suspends, the others run), the kernel keeps to one warp per call, and waiting branches are emitted after their siblings |
| `cudaFreeHost` of a pointer it did not allocate crashed the program | it called `free()` on anything | it only frees what `cudaMallocHost` allocated |

### Missing features (the programs did not compile)

| Feature | Fix |
|---|---|
| Atomics (`atomicAdd`... `atomicCAS`, `__threadfence`) | declared in the device header; the SIMT pass runs one locked instruction per active lane, or, when every lane updates one address, one reduction across the lanes and a single atomic, with an exclusive scan for the returned values |
| `__device__` and `__constant__` variables, `cudaMemcpyToSymbol` | the variables live in the VRAM, one copy per program, reached by every module (the generic one and each block shape's) through a pointer the runtime sets at load time |
| `volatile` accesses | kept in order behind a compiler barrier |
| `__syncthreads_count`, `_and`, `_or` | a counter in shared memory around an ordinary barrier (three in turn) |
| Device `printf`, `malloc`/`free`/`new`, `assert` | one call per active lane into the process's own functions; `printf` output goes to the program's standard output (passed to the driver when it connects) |
| `cooperative_groups.h`, `<cuda/atomic>` | headers over the warp primitives and the atomic builtins; the grid barrier works in cooperative launches |
| CUDA Graphs | stream capture and explicit nodes, run in the order they were added |
| `__managed__` variables, mapped `cudaHostRegister` | kept equal to the host's copy around each launch |
| About 40 runtime API functions | `cudaDeviceGetAttribute`, pitched and 3D memory, function attributes, occupancy, pointer attributes, stream flags, limits... |
| About 120 math functions and intrinsics | directed rounding (`_rz/_ru/_rd`), conversions, `frexp`, `remquo`, `erfinv`, `normcdf`, Bessel... |

### Wrong results and compiler errors

| Symptom | Cause | Fix |
|---|---|---|
| 12 single-precision functions (`expf`, `logf`, `cbrtf`...) over CUDA's ulp bound | libmvec's float versions err by up to ~4 ulp | computed in double (libmvec's double versions) and rounded once |
| Double-precision functions over their bound | libmvec's double versions err by up to ~4 ulp, CUDA's by 1 or 2 | the C library's, once per lane; `cbrt`, `rsqrt`, `norm3d` refined with one Newton step on two-double residuals |
| `sinpif` 13 million ulp off for large x | implemented as `sinf(pi * x)` | exact reduction, then pi times the remainder as a sum of two doubles |
| "internal: no scalar value" on struct copies through pointers | a `float4` copy is an `llvm.memcpy`, expanded only next to local arrays | every `memcpy`, `memmove` and `memset` becomes loads and stores |
| Integer division by zero, or `INT_MIN / -1`, killed the kernel | x86's `idiv` faults where a GPU returns an unspecified value | those divisors are replaced by 1 |
| Recursive functions taking a local array (quicksort) did not compile | per-thread arrays are laid out lane-major, with no per-lane address | such arrays get W real copies |
| `__reduce_*_sync` with a partial mask, `__sad` with a large difference, `normcdfinvf(0.5)` returning -0 | implementation errors in the device header | fixed |

## What is left

- `j0` and `y0` (double): 2 cases. glibc's functions err by about 10 ulp near their zeros, CUDA's bound is 7.
- A kernel of a cooperative launch with more blocks than the device runs at once is refused (`cudaErrorCooperativeLaunchTooLarge`), as on a GPU; here that is two blocks per multiprocessor.
- Kernels that wait for other threads run one warp per call, which is slower, and a lane waiting for another lane of the same warp still hangs (as on GPUs before Volta).
- Mapped and managed memory are coherent at kernel boundaries only, not while a kernel runs.

## The slow cases, for a later round

`kernel_ms / ref_ms` compares the CUDA program on the simulator with plain C++ and OpenMP on the same CPU, measured in the same run. Of the 113 cases whose OpenMP version takes 1 ms or more, 48 are slower on the simulator, 30 by more than 1.5x, and 36 are at least 2x faster. Grouped by what they have in common (the causes are hypotheses to verify, not measurements yet):

| Group | Cases (ratio) | What they share |
|---|---|---|
| Many short launches | `smith_waterman_diag_launches` (124x, 2047 launches), `floyd_warshall_k_launches` (8.3x, 1024), `lcs_tiled_wavefront` (9.2x, 255), `jacobi2d_20iters` (4.1x) | about 66 us per launch through the driver, against 2 us on the GPU |
| Global atomics with conflicts | `histogram_global_atomics` (522x, 156 ms), `hist256_global_atomics` (124x), `unsigned_hist64`, `minmax_unsigned`, `and_or_xor`, `double_bins_tol` | one locked instruction per lane, on lines that every core writes |
| Sorting networks and scans | `bitonic_sort_1M` (7.2x), `odd_even_merge_sort_1M` (8.1x), `bitonic_sort_dups_512K` (11.7x), `radix_sort_1M` (6.2x), `stream_compaction` (13.3x), `large_scan_3kernels` (4.5x), `blelloch_block_cfo` (3.1x) | barriers in loops, data-dependent shared-memory indices |
| Bytes and irregular access | `rle_decode_bsearch` (22.5x), `sobel_u8` (8.3x), `rle_decode_scatter` (6.3x), `base64_pad1` (3.3x), `crc32_*` (2.4 to 3x) | byte loads and stores per lane (AVX-512 has no byte gather), table lookups |
| Divergence and per-thread memory | `array_by_pointer_quicksort` (5.1x), `early_return_nested_loops` (3.3x), `local_4kib` (1.7x) | per-lane recursion depths, arrays with real per-lane addresses |
| Other | `matmul_regblock4x4` (2.8x), `spmv_csr_scalar` (2.7x), `pagerank_20_iters` (2.3x), `gauss_separable_r8` (2.0x) | |

The full list is the suite's `summary.tsv`.

Previous: [28](28-function-calls.md). Next: [30](30-launches-and-hot-spots.md), which worked on these slow cases. Back to the [iteration 3 index](README.md).
