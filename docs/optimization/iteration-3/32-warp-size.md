# 32. The warp size: 16, 32 or 64 threads, and blocks smaller than a warp

> **In one sentence:** `CPUSIM_WARP_SIZE=16` or `64` at compile time changes the warp of a whole program (CUDA's 32 stays the default), and independently of it a block smaller than a warp now runs in fewer lanes (a block of 16 threads in one 16-lane vector instead of a 32-lane warp with half its lanes masked): kernels with blocks of 8 and 16 threads 10 to 46% faster; warps of 16 help divergent loops and hurt regular memory-bound and barrier kernels, about even on the whole, so 32 stays the default.

The general concepts (lanes, warp entry, variants) are in [00-basic-concepts.md](../00-basic-concepts.md); the variants and how the runtime picks one are in [02](../iteration-1/02-width-variants.md) and [03](../iteration-2/03-automatic-width-selection.md).

## The problem

The warp entry, the function that runs one warp, always had 32 lanes: two zmm registers per 32-bit value. Two things follow:

- **A block smaller than a warp wastes lanes.** A block of 16 threads runs as a warp of 32 with lanes 16 to 31 masked off, and every instruction still processes both registers. At a barrier, the executor switches between groups that are half empty.
- **The warp size is a semantic constant.** Divergence is decided per warp (a vector of 32 executes both sides of an `if` if any of its 32 lanes takes each), and a barrier suspends groups of 32. A CPU register holds 16 values: code written against `warpSize` rather than a literal 32 could run with warps of 16, diverging and suspending at a finer grain. AMD GPUs use 64 (a wavefront).

## The warp size of a program

`CPUSIM_WARP_SIZE` is read by `cucc`, which passes it to Clang as `__CPUSIM_WARP_SIZE__`. Everything that depends on it follows from there:

1. **The device header** defines `warpSize` from it, and with it the default width of the shuffles, `__reduce_*_sync`, `__match_*_sync` and cooperative groups (tiles up to 32 and up to `warpSize`).
2. **The module.** The header also leaves `__cpusim_warp_size` in the device bitcode. `cpusim-devc` reads it and removes it, so the code compiled ahead of time and the JIT (which compiles the same bitcode) agree without passing anything else. The compiler derives the lane id (`lane % warpSize`), the shuffles (a permute within each warp's 1, 2 or 4 vectors of 16) and the ballots from it, and builds the variants in units of warps: the wide ones with `2 x warpSize << i` lanes up to 512, the suspending ones with 2 and 4 warps.
3. **The runtime.** The kernel metadata carries the warp size and the warp entry's lanes (kernel ABI 13), and the manifest the warp size: the executor groups threads by it, and `cudaGetDeviceProperties` and `cudaDevAttrWarpSize` report it. Modules of one program compiled with different warp sizes are refused.

### Masks with warps of 64

CUDA's warp masks are `unsigned`: 32 bits. With warps of 64:

- `__ballot_sync`, `__ballot` and `__activemask` return lanes 0 to 31 only;
- a mask argument of `0xffffffff` means the whole warp;
- `__any_sync`, `__all_sync`, `__uni_sync` and cooperative groups (tile ballots, `match_any`, coalesced groups) work on all 64 lanes, through an internal 64-bit ballot (`__cpusim_ballot`, which the compiler lowers).

HIP code that stores 64-bit masks needs HIP's headers, which `cucc` does not provide.

### What does not change

- Tensor cores (`wmma`) need warps of 32: a kernel that uses them is a compile error with another size.
- Code that assumes 32 (a reduction starting at `__shfl_down_sync(mask, v, 16)`, a tile of 32 with warps of 16) is wrong with another size. That is the reason 32 stays the default; 16 is for code written against `warpSize`.

## Blocks smaller than a warp

With the block shape known (the module compiled for a constant shape, or the JIT), the compiler halves the warp entry's lanes while the block fits in half: a block of up to 16 threads gets a 16-lane entry (up to 32 with warps of 64). The warp keeps its semantics: lane ids, `warpSize` and the ballots are those of the full warp, and the lanes left out are threads that do not exist (a shuffle from one of them returns the caller's own value; CUDA leaves it undefined). The metadata's `warp_lanes` tells the executor the group size. Tensor core kernels keep 32 lanes.

Measured with blocks of 8 and 16 threads (16M threads, same driver, three alternating rounds, results unchanged):

| Kernel | 32 lanes (ms) | Narrowed (ms) | Speedup |
|---|---|---|---|
| 64-step polynomial, blocks of 16 | 6.05 | 5.46 | 1.11x |
| same, blocks of 8 | 11.57 | 10.24 | 1.13x |
| shared-memory sum with a barrier, blocks of 16 | 13.16 | 9.04 | 1.46x |
| same, blocks of 8 | 15.42 | 11.36 | 1.36x |

## Warps of 16 against 32

Two measurements on the same driver, programs compiled once per size: a first one (two rounds of the six `bench/` kernels and 11 stress programs, each size's run in turn) and, for the cases that moved most, a second one interleaving the two sizes program by program over three rounds. Only the cases that moved the same way in both are trusted; others (the `bench` matmuls: 0.69 then 1.18, `black_scholes`: 0.71 then 1.22) flipped, which is clock and ordering noise. Time with warps of 16 over time with 32, both measurements:

| Case | First | Second | |
|---|---|---|---|
| `spmv_csr_scalar` | 0.71 | 0.76 | a loop per row, of a different length per thread |
| `mandelbrot` (bench) | 0.87 | 0.88 | per-pixel iteration counts |
| `lcs_tiled_wavefront` | 0.88 | 0.88 | |
| `nbody_allpairs_16384` | 0.87 | 0.90 | |
| `scan_blocks` (bench) | 0.93 | 0.90 | |
| `saxpy_grid_stride` | 1.15 | 1.12 | regular, memory-bound |
| `dequant_int8_per_channel` | 1.52 | 1.20 | |
| `smith_waterman_diag_launches` | 1.28 | 1.25 | barriers in small blocks |
| `floyd_warshall_blocked_shared` | 1.17 | 1.33 | |
| `matmul_naive` (stress) | 1.54 | 1.63 | |

Over the second measurement's 27 cases, the geometric mean is 1.05 (8 at least 5% faster with 16, 12 at least 5% slower). Warps of 16 pay off where lanes diverge (a vector of 16 waits for fewer slow lanes) and cost where the code is regular: half the lanes per call of the warp entry, twice the calls and the switches at barriers, and wider variants that the selection does not always pick. Hence 32 stays the default, and 16 is a per-program choice, for code written against `warpSize` that diverges.

The whole stress suite rebuilt with each size (631 cases): with 16, 591 pass; 32 fail, one program does not compile (`sync_coop_groups` asks for `tiled_partition<32>`, which a warp of 16 cannot hold) and one faults (`wl_sort`'s radix scatter computes its scan with `threadIdx.x & 31`, so its indices go out of range). With 64, 609 pass, 17 fail and 2 hang (`sync_locks`, `sync_barriers`: locks between two 32-thread warps become locks inside one 64-thread warp, which a SIMT machine without independent thread scheduling cannot run). Every failure is in code that assumes 32 (`threadIdx.x & 31`, `>> 5`, reductions starting at `__shfl_down_sync(mask, v, 16)`, `0xffffffff` as a full 64-lane mask, and `sync_warp`, which tests the 32-thread semantics on purpose); it would fail on a GPU with that warp size too. No compiler error is left.

## Validation

`tests/cuda/warp_size.cu`, built with each warp size (`cuda_warp_size_16`, `_32`, `_64` in CTest) and run in every execution mode the feature tests use: warp sums and scans with shuffles, tile maxima with `cg::reduce`, `__any_sync`/`__all_sync`, tile ballots and `match_any`, lane ids, `warpSize` on the device and in `cudaDeviceProp`, blocks of 8 and 16 threads (narrowed entries), and blocks of 7, 100, 3 warps + 5 and 1000 threads with a barrier. The 32-thread build also passes on the GPU with `nvcc`. The rest of CTest and the stress suite (631 of 633, the two left being `j0` and `y0`) pass unchanged with the default size.

Two fixes along the way. Three loops over the active lanes (atomics grouped by address, per-lane external calls, calls through a pointer) turned the index of the first lane, a W-bit count, into an `i32` with a truncation, which is invalid IR when W is 16; it is now a zero extension or a truncation as needed (`wl_graph` and `wl_sort` did not compile with warps of 16). And the clean-up pipeline after the SIMT pass ran `instcombine` with LLVM's fixpoint verification, which aborted on the 16-lane entries ("Instruction Combining did not reach a fixpoint"); every other pipeline of `cpusim-devc` already turned it off ([11](../iteration-1/11-llvm-tuning.md)).

Previous: [31](31-gathers-and-scatters.md). Next: [33](33-xeon-server.md). Back to the [iteration 3 index](README.md).
