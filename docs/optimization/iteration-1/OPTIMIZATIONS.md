# CPU execution optimizations

> This is the original record of **iteration 1** of the optimizations, kept as it was written. Each optimization here has a detailed document in this folder (see the [iteration README](README.md)); the following optimizations are in [iteration 2](../iteration-2/README.md).

A record of the optimizations applied to the compiler (`cpusim-devc`), the runtime (`libcpusim_cudart`) and the driver's executor, with the problem each one solves, how it works and where it is in the code. The numbers come from `bench/run.sh` (kernels only, sample workload, result checked bit for bit against the RTX 5060).

## Overall result

| Kernel | RTX 5060 | CPU before | CPU after |
|---|---|---|---|
| matmul_naive | 163 ms | 2400 ms | 224 ms |
| matmul_tiled | 126 ms | 3179 ms | 304 ms |
| scan_blocks | 20.6 ms | 1072 ms | 399 ms |
| bitonic_sort | 26 ms | 1099 ms | 709 ms |
| mandelbrot | 95 ms | 797 ms | 715 ms |
| transpose naive / tiled | 15 / 11 ms | 149 / 165 ms | 142 / 135 ms |
| heat_step | 1.8 ms | 20.1 ms | 15.6 ms |

---

## How to measure (before optimizing anything)

### Kernel-only benchmark (`bench/`)

Each file in `bench/` includes the sample's `.cu` with `main` renamed (`#define main sample_main`), so the kernels are exactly those of the sample, but without the CPU reference (which took up to 2 minutes). It runs 1 warm-up + 5 repetitions and reports the best time.

- `bench/run.sh both|gpu|cpu [names]` compiles with `nvcc` and with `cucc` and runs both.
- Each benchmark prints a hash of the output (`CHECK`). Since the build uses `-fmad=false`, GPU and CPU produce bit-for-bit identical results, and the script compares the hashes (`OK` / `DIFF`). Every optimization was validated this way.

### Cycles in addition to time

The laptop changes its clock with temperature, and the same kernel varied by up to 2x between runs. The harness opens a cycle counter with `perf_event_open` (with `inherit`, to include the executor threads created later) and reports `Gcyc` alongside the time. Even cycles vary in memory-bound code (DRAM latency in cycles depends on the clock), so fine decisions were made by comparing variants **in interleaved pairs in the same process** (naive, tiled, naive, tiled...).

### Tools

- `perf record` / `perf annotate` (requires `kernel.perf_event_paranoid <= 1`).
- Zen 4 **IBS** (`perf record -e ibs_op/cnt_ctl=1/`): precise per-instruction sampling, without the *skid* of ordinary sampling. It is what showed the real distribution of operations.
- `perf stat` with instructions, IPC and L1 misses to separate "too many instructions" from "stalls".
- So that perf could find the symbols, the kernel modules came to be loaded from a content-addressed file (`$XDG_RUNTIME_DIR/cpusim/modules/<hash>.so`) instead of an anonymous memfd (`src/device/device_core.cpp`, `module_file`).
- `cucc --keep` keeps the per-thread IR (`.thread.ll`), and the vectorized IR before (`.raw.ll`) and after (`.warp.ll`) O3.

---

## 1. JIT at launch, specialized by block shape

**The problem.** The compiler does not know `blockDim`: `threadIdx` comes from memory at run time. Therefore every memory access needs run-time tests to discover the pattern (contiguous? same address?), and addresses that would be `base + constant` become one pointer per block of 16 lanes, which exhausts the registers in the wide variants.

**How it works.**
- In addition to the generic `.so`, `cucc` embeds in the fatbin the device's **per-thread bitcode** (the role of PTX). The fatbin header gained `bitcode_offset/size` (ABI 6, `src/device/kernel_abi.h`).
- At the first launch of each block shape, `libcpusim_cudart` writes out the bitcode, calls `cpusim-devc --specialize X,Y,Z` and links the `.so`. The result is kept in `~/.cache/cpusim/jit/<bitcode hash>-XxYxZ.so`, so later runs of the program do not recompile. Typical compilation: 0.5 s.
- With the shape known (`emit_specialized_tid` in `src/compiler/simt.cpp`):
  - `blockDim` becomes a constant;
  - when the group covers the whole block, `threadIdx` becomes a **constant vector**;
  - when the width is a multiple of `blockDim.x` and the group does not cross z planes, `threadIdx.x` is constant and `y`/`z` are a constant + a uniform value derived from the warp index;
  - the mask of valid lanes becomes a constant when the block is a multiple of the width.
- With this, LLVM resolves the pattern tests at compile time and the accesses become broadcasts and loads with an immediate displacement. The inner loop of the tiled matmul became `vmulps -0x3c0(%rcx){1to16}, ...` with 16 accumulators.
- In specialized mode, only the variants that the runtime may choose are generated, and no instrumented one (profiling uses the generic module).
- If the JIT fails or is disabled (`CPUSIM_JIT=0`), the generic module is used.

**Impact.** The largest isolated gain: matmul naive from ~313 to 280 ms and tiled from ~600 to ~390 ms with this alone, before the address decomposition fixes that came later.

---

## 2. Variable width of the thread group and choice by measurement

**The problem.** A warp of 32 lanes per call has little independent work. The matmul loop accumulates `acc += a*b` in sequence, and the sum cannot be reassociated (the result must be bit for bit equal to the GPU's). With 2 dependency chains (32 lanes = 2 zmm), the CPU waits on the latency of the addition. Kernels with a barrier also pay one context switch per warp at each `__syncthreads`.

**How it works.**
- The width `W` became a parameter of the code generator. The call context (`cpusim_warp_ctx`) has a 128-byte header followed by the per-lane data of the variant's size.
- Generated variants:
  - **32 lanes**: one warp, suspends at barriers (always exists);
  - **64 and 128 lanes** for kernels without a barrier: 2 or 4 warps per call, more independent chains;
  - **whole block (64 to 512 lanes)** for kernels with a barrier: the whole block advances together and `__syncthreads` becomes nothing (no frame, no resume);
  - **suspended 64s and 128s**: groups of 2 or 4 warps that still suspend at barriers.
- The CUDA warp semantics remain 32 (`laneid = lane % 32`, `warpSize = 32`).
- **Autotuning** (`pick_width` / `record_timing` in `src/device/device_core.cpp`): in the first launches of each (kernel, block shape), the runtime tries the valid widths, measures the time per block and keeps the best. The first candidate is measured again at the end (the first launch pays for cold cache and page faults), and launches too small to occupy the workers do not count. `CPUSIM_LANES=<n>` or `<n>s` forces a width; `CPUSIM_VERBOSE=1` shows the measurements.

**Impact.** Typical choices: naive 128 lanes, tiled 256 (whole block), transpose naive 128, scan and bitonic 32 or 64s (heavy divergence favors small groups, which skip warps without active lanes).

---

## 3. Block order in bands (swizzle)

**The problem.** The 28 workers took consecutive blocks in row order. The simultaneous blocks shared the rows of A, but each one read different columns of B, and B (67 MB) does not fit in the L3, so it was reread from DRAM at each band of blocks.

**How it works.** The executor (`run_blocks` in `src/device/executor.cpp`) maps the linear block counter to bands of 8 columns walked row by row, like the *thread-block swizzle* used on GPUs. The blocks in execution cover ~8 columns x ~3.5 rows and reuse A and B in the cache. CUDA does not guarantee block order, so it is safe. `CPUSIM_BLOCK_ORDER=linear` turns it off.

**Impact.** −27% on naive and −28% on tiled.

---

## 4. Decomposing the address into uniform base + varying offset

**The problem.** A load through a varying pointer computed 32 64-bit pointers and tested the pattern over them at each iteration.

**How it works** (`split_address` / `split_index`). The per-thread pointer is rewritten, looking through GEPs, as `uniform base + varying byte offsets`. The uniform part (for example the loop's `k`) stays scalar. The varying part usually depends only on `threadIdx`, so it is loop-invariant and everything computed from it leaves the loop. The decomposition distributes over:
- `add` with `nsw`/`nuw` according to the extension (sext/zext);
- `or disjoint`, which is what instcombine makes of `t + threadIdx.x` when the bits do not overlap;
- `add` **without flags under sign extension**. CUDA mixes `int` with `unsigned` `threadIdx`, so the addition comes out without a flag. It only differs from the exact one if the signed sum leaves the `int` range, and in that case the original index would already be ~2^31 elements outside any allocation. Under zero extension this is not done, because there the overflow may be legitimate;
- `mul` and `shl` by a uniform value (`(t + ty) * N` becomes `t*N + ty*N`).

---

## 5. Dynamic coalescing of gathers and scatters

**The problem.** Generic gather and scatter are expensive on Zen 4 (the scatter is microcoded), but most CUDA accesses are contiguous or to the same address per group of lanes.

**How it works** (`gather` / `scatter`). Per block of 16 lanes (one zmm), the pattern is decided from the offsets:
- **contiguous**: one masked load/store;
- **same address**: one scalar load + broadcast;
- otherwise, a real gather/scatter (which also preserves "the last lane wins" when lanes write to the same address).

Inactive lanes are never read: masked loads only touch active lanes, and the tests only look at active lanes.

Refinements applied later:
- **Test bits in integer registers.** The invariant test is kept as `i16` and combined with the mask in scalar. Before, the codegen recomputed the `vpcmpneqq` at each iteration for lack of mask registers (there are only 8). This adjustment took naive from 2340 to 1265 ms.
- **Test the whole access first.** Before the per-block-of-16-lanes tests, it tests patterns of the whole access: all contiguous with the same start (`Bs[k][tx]` in a 16x16 block, a single load replicated to all blocks), all contiguous (one load per block, no tests) and all single-address (one broadcast per block). The rest goes to the per-block path.
- **Broadcast without a branch.** A block with no active lanes may have an invalid address. Instead of branching, the address becomes `select(any_active, address, context)` and the (discarded) read happens in the context, which is always readable.
- **Offsets only on the rare path.** The 64-bit offset vectors are only rebuilt inside the generic gather path, so the common path does not keep them live.

---

## 6. Uniform loops with real control flow

**The problem.** Linearization treated every branch with masks: each block carried the mask, tested whether there was an active lane and propagated masks to the successors. This applied even to the loop `for (k = 0; k < K; ++k)`, which is uniform.

**How it works** (`is_uniform_loop` / `emit_uniform_loop`). If all the branches of a loop (exits, return to the header, internal branches and subloops) are uniform, all active lanes follow the same path. The loop is then emitted with its original CFG and scalar branches, and the entry mask holds for all blocks. There is no per-block mask, no per-iteration test, and no reinitialization of masks. Loops with a barrier inside are also accepted: the mask stays in the header's slot (saved and restored with the rest) and is reread in each block.

**Impact.** Naive from 912 to 597 ms when it landed.

---

## 7. SIMT if-conversion with guarded loads

**The problem.** Patterns such as `As[ty][tx] = (row < M && a_col < K) ? A[...] : 0` became a linearized divergent branch: 3 blocks, masks, a slot with a blend and tests. In the experiment without a barrier, this alone cost ~265 ms in tiled.

**How it works** (`if_convert`). A side of a divergent branch that is small (up to 24 instructions), has no stores, calls or barriers, and contains only arithmetic that cannot raise an exception and loads, is moved to the parent block. The loads become calls `__cpusim_cond_load.<T>(p, c, other)`, which in vectorization are masked loads with `mask & c`. The phis at the join become `select`s. Divisions are excluded, because they could raise an exception in lanes that would not execute. The instrumented variant (profiling) does not go through this, to keep the per-thread counts exact.

**Impact.** Tiled without a barrier went from 760 to 525 ms, tying with naive.

---

## 8. Barriers: saving only what is needed, and cheaper

A warp suspended at a barrier writes state into its frame and resumes later through a dispatcher. The following optimizations reduced what is written and what is recomputed.

### 8.1 Liveness at the resume points

Only the slots that may be read before being rewritten after the barrier are saved (`compute_resume_liveness`), and each resume point has its own restoration.
- **Value slots** use liveness over the **per-thread** code (`compute_thread_liveness`). Each lane follows its own path, linearization preserves per-lane order and a masked blend fully defines the slot for the lane that executes it. The same analysis over the linearized code would be wrong, because it does not see that a block skipped for lack of lanes also skips whoever reads it.
- **Masks** (which exist only in the linearized code) use liveness over it, modeling the barrier's `return` as an edge to its resume point. The absence of this edge was a **latent bug**: values live across two consecutive barriers might not be saved.

### 8.2 Rematerialization of invariants

Values that are pure functions of `threadIdx`, `blockIdx`, parameters and constants (such as `row`, `col` and the pointers `&As[ty][tx]`) get no slot: they are cloned into the block where they are used (`rematerialize_invariants`). This removed most of the state from the frame (from ~25 vectors saved per barrier to the accumulator and a few scalars).

### 8.3 Fewer duplicated slots

- Single-input LCSSA phis are folded when the uniform/varying classification matches that of the source.
- The reload of a varying slot used in another block is redone at the use when no per-lane path between the two writes to the slot (`reload_is_safe`), instead of getting a copy slot with a blend.

### 8.4 Per-warp invariant cache in the frame

Data derived only from the context (the start of each block of 16 lanes and the pattern bits of each access) are computed once, when the warp starts (block `warp.invariants`), written to the frame and reread as scalars where needed, including after each resume (`cached`). Since they never change, they do not need to be saved at barriers.

### 8.5 Layout

- The context was compacted: a 128-byte header followed by the per-lane data of the width used. Before, it was 6.3 KB per warp, with fields 2 KB apart that fell into the same L1 sets.
- Consecutive frames and contexts are an odd number of cache lines apart (`cpusim_stagger`).
- The frame size is recorded per variant.
- Each worker's shared buffer starts at a different offset within the page, because SMT siblings share the L1.

### 8.6 Invariant reads

Reads of the context and of the parameters receive `!invariant.load` (and `dereferenceable`), so the register allocator can reload them instead of spilling. The frame is left out, because it changes between calls.

---

## 9. Making more use of LLVM

- `-unswitch-threshold=2000` (default in `cpusim-devc`): lets LLVM version loops by the invariant access-pattern flags, taking the tests out of the loops. `CPUSIM_LLVM_FLAGS` passes other options for experiments.
- `instcombine<no-verify-fixpoint>` in the pre-vectorization pipeline: the LLVM 19 textual pipeline aborts if instcombine does not converge in one pass.

---

## 10. Prefetch support

`llvm.prefetch` (from `__builtin_prefetch`) with a varying pointer becomes one `prefetcht0` per block of 16 lanes, at the start of each block, with the same safe-address trick for inactive blocks. There is no automatic insertion yet; the idea is to use SCEV on the per-thread code to prefetch the next iteration of global loads in loops.

---

## What was tried and did not help

They are recorded so as not to be repeated:

- **Block function** (`__cpusim_block_<kernel>`: the block's barrier rounds inside the generated code, with the warp inlined). It was slower than the per-warp calls, because of register pressure in the large function. It is still generated and is used only on the 32-lane path; `CPUSIM_NO_BLOCK_FN=1` turns it off.
- **Whole-block variant without specialization.** Without the JIT, it is worse than the 32-lane one: with 16 pointers per access and per-block tests, registers run short. With the JIT, it became the best for tiled.
- **Executing the warps of a block on several cores.** It was not implemented: with tens of thousands of blocks, all cores are already busy, and splitting a block would add a barrier between cores.

## Pending items

- **Tiled:** in cycles, it is still ~1.15x of naive. The tile load in the 256-lane variant still weighs.
- **`scan` and `bitonic`** (20 to 27x of the GPU): divergent branches with stores near the barriers. None of the optimizations above reaches this case.
- **Automatic prefetch** guided by SCEV.
- **`mandelbrot`:** a pure divergent loop, 7.5x of the GPU. It may benefit from compacting active lanes across warps.
