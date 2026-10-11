# 00. Basic concepts

This document gathers the vocabulary and mechanisms that the other documents use. It does not explain undergraduate basics (what a cache, a thread or a compiler is); it explains the specialized parts that not every computer science graduate has seen up close: the CUDA execution model, how it is mapped to SIMD on the CPU, AVX-512 and its mask registers, the Zen 4 memory hierarchy, and the LLVM pieces that appear in the optimizations.

## The machine

All measurements were made on a laptop with:

| Component | Detail |
|---|---|
| CPU | AMD Ryzen 9 8940HX (Zen 4, "Dragon Range"): 16 cores, 2 SMT threads per core (32 logical CPUs) |
| Vectors | AVX-512 (F, BW, DQ, VL and other extensions). On Zen 4, most 512-bit operations are executed as two 256-bit halves in the floating-point units |
| Per-core caches | 32 KiB L1 data, 1 MiB L2 |
| L3 | 64 MiB in total, in **two CCDs** (chiplets) of 8 cores with 32 MiB each. A core directly uses only the L3 of its own CCD; data in the other CCD goes through the Infinity Fabric |
| Memory | DDR5 on two channels. The bandwidth measured with babelstream (Triad) was ~33 GB/s |
| Reference GPU | NVIDIA RTX 5060 Laptop (compute capability 12.0), used to validate results and compare performance |

The simulated "device" reserves 14 of the 16 cores (logical CPUs 2 to 15 and 18 to 31); cores 0 and 1 (CPUs 0, 1, 16 and 17) are left for the host process and the system.

## The CUDA execution model

A **kernel** is a function executed by many threads. The launch `kernel<<<grid, block>>>(args)` defines a **grid** of **blocks**, and each block has a fixed number of threads (up to 1024). Each thread knows:

- `threadIdx` (x, y, z): its position within the block;
- `blockIdx` (x, y, z): the position of its block in the grid;
- `blockDim` and `gridDim`: the dimensions of the block and of the grid.

The typical global index is `blockIdx.x * blockDim.x + threadIdx.x`. The limits that CUDA guarantees (and that some optimizations use to prove that indices do not overflow) are `gridDim.x < 2^31` and `gridDim.y, gridDim.z < 65536`.

Guarantees and non-guarantees that matter to us:

- **Blocks are independent.** CUDA guarantees neither execution order nor simultaneity among blocks; they communicate only through global memory between launches (or through atomics, which this project does not support yet). For that reason the runtime may execute the blocks in whatever order and on whatever core it wants.
- **The threads of a block cooperate** through **shared memory** (memory visible only to the block, declared with `__shared__`) and synchronize with `__syncthreads()`, a barrier: no thread of the block passes it before all have arrived.
- **Warps.** The hardware groups the threads of a block into **warps** of 32 consecutive threads (in the linear order `threadIdx.x + threadIdx.y * blockDim.x + ...`). The threads of a warp execute in lockstep most of the time, which gives meaning to the warp operations (`__shfl_sync`, `__ballot_sync`). Since Volta, the GPU has *independent thread scheduling*: threads of a warp can diverge and reconverge, and `_sync` functions require all threads in the given mask to participate (which is why a `__ballot_sync(0xffffffff, ...)` inside a divergent `if` hangs the real GPU).
- **Coalesced access.** On the GPU, if the 32 threads of a warp access consecutive addresses, the hardware merges everything into a few transactions. Well-written CUDA code is full of such accesses, and that works in our favor: consecutive addresses between neighboring threads are also the good case on the CPU.

## How CUDA becomes CPU code: SIMT on top of SIMD

The GPU is a **SIMT** machine (*single instruction, multiple threads*): the programmer writes the code of **one** thread, and the hardware executes many in parallel. The CPU with AVX-512 is **SIMD**: one instruction operates on a vector of values. The project's compiler (`cpusim-devc`) bridges the two: **each CUDA thread becomes a *lane* of a vector**.

- A `zmm` register has 512 bits: 16 32-bit floats or 8 doubles.
- A warp of 32 threads occupies 32 lanes, that is, 2 `zmm` registers per float variable.
- The generated code processes **W lanes per call**. W = 32 corresponds to one warp; the wide variants process 64, 128, 256 or 512 lanes (several warps, or the whole block) in a single call. See [02](iteration-1/02-width-variants.md).

Example with a toy warp of 4 lanes: the per-thread code

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
out[i] = a[i] * 2.0f;
```

becomes, for block 0 with `blockDim.x = 4`:

```text
threadIdx.x = [0, 1, 2, 3]          (constant vector, with the JIT)
i           = [0, 1, 2, 3]
a[i]        = contiguous load from a[0..3]
out[i]      = contiguous store to out[0..3] of 2 * a[0..3]
```

One vector instruction does the work of the 4 threads. The real implementation is in `src/compiler/simt.cpp` (class `WarpBuilder`).

### Uniform and varying

The **uniformity analysis** (`src/compiler/uniformity.cpp`) classifies each value in the per-thread code:

- **uniform:** equal in all lanes of a call (kernel parameters, `blockIdx`, `blockDim`, constants, loop counters that do not depend on `threadIdx`). It lives in an ordinary scalar register, computed once;
- **varying:** may differ between lanes (`threadIdx`, everything that depends on it, loads from a varying address, values that come out of loops with a divergent exit). It lives in a vector.

The distinction is central: a uniform value costs one scalar instruction, a varying one costs one instruction per vector register, and a varying **address** may require a gather.

### Divergence and linearization

When lanes of a warp take different sides of an `if`, the CPU cannot follow two paths at the same time. The compiler **linearizes** the flow: it executes both sides in sequence, each with a **mask** of active lanes, and combines the results (blend). Example with 4 lanes and `if (x > 0) y = a; else y = b;`:

```text
x            = [ 5, -1,  3, -7]
mask then    = [ 1,  0,  1,  0]   -> computes a in lanes 0 and 2
mask else    = [ 0,  1,  0,  1]   -> computes b in lanes 1 and 3
y            = [a0, b1, a2, b3]
```

**Uniform** branches (all lanes go to the same side) do not need this and become ordinary scalar branches. Loops with a divergent exit keep iterating while some lane is active. See [07](iteration-1/07-uniform-loops.md) and [08](iteration-1/08-if-conversion.md).

### Barriers on the CPU

In the 32-lane variant, a block of 256 threads is 8 calls (one per warp). At a `__syncthreads`, the warp cannot continue without the other 7: it **writes its live state** into a *frame* (memory owned by the warp) and returns to the executor, which runs the other warps up to the same barrier and then calls each one again; a dispatcher at the start of the function restores the state and continues after the barrier. In the whole-block variants, the 256 threads are in the same call and the barrier costs nothing. See [09](iteration-1/09-state-at-barriers.md) and [10](iteration-2/10-barrier-pass.md).

## AVX-512 essentials

- **Registers:** 32 `zmm` registers of 512 bits (`zmm0` to `zmm31`).
- **Mask registers:** 8 registers `k0` to `k7`, of up to 64 bits, one bit per element. Almost every instruction accepts a mask: `vaddps zmm1{k1}, zmm2, zmm3` only writes the elements with bit 1 (with `{z}`, it zeroes the others). `k0` as a mask means "no mask", so in practice there are 7 predicate registers. Vector comparisons (`vpcmpd`, `vcmpps`) produce masks. This is how linearization with masks becomes cheap code, and also why too many masks cause pressure on a few registers.
- **Embedded broadcast:** `vmulps zmm0, zmm1, [rax]{1to16}` multiplies by a scalar read from memory and replicated into the 16 positions, with no extra instruction.
- **Masked load and store:** `vmovups zmm0{k1}{z}, [rax]` reads 16 consecutive floats only in the active positions, without page faults on the inactive ones. It is the cheap way to read 16 threads with consecutive addresses.
- **Gather and scatter:** `vgatherdps`/`vgatherqps` read 16 (or 8) arbitrary addresses; `vscatterdps` writes to them. On Zen 4 they are much more expensive than a contiguous load (the gather is split into individual accesses; the scatter is microcoded), and each element may touch a different cache line.
- **Permutations:** `vpermi2d` builds a 16-element vector by choosing freely among 32 (two registers) according to an index vector. It is the basis of `__shfl_sync`.

Because Zen 4 executes 512 bits as two halves, the advantage of AVX-512 on it comes less from width and more from masks, embedded broadcast, gathers and permutations.

## Memory hierarchy and what it costs

| Level | Size (per core, unless noted) | Approximate latency |
|---|---|---|
| L1 data | 32 KiB | ~4 to 5 cycles |
| L2 | 1 MiB | ~14 cycles |
| L3 | 32 MiB per CCD (shared by 8 cores) | ~50 cycles |
| DRAM | | ~100 ns (hundreds of cycles) |

- **Cache line:** 64 bytes. Reading one float brings in the 16 neighboring floats. An aligned contiguous load of 16 floats touches one line; a gather of 16 scattered addresses may touch 16 lines.
- **Latency versus bandwidth:** a kernel may be limited by **bandwidth** (bytes per second that DRAM delivers, ~33 GB/s here) or by **latency** (how many accesses are in flight at the same time). HeCBench's matrix-rotate is the example of the second category: more cores, or prefetch, put more accesses in flight. See [19](iteration-2/19-automatic-prefetch.md).
- **CCDs:** the L3 is private to each CCD. Data read by a core of CCD 0 is not in the L3 of CCD 1. For kernels that run many times over the same data (time steps of a stencil), keeping each block always on the same core keeps the data in the right L3. See [17](iteration-2/17-block-distribution.md).
- **Hardware prefetchers:** Zen 4 has prefetchers that detect sequential streams and constant-stride streams, learning by the address of the instruction that makes the access. They cover well a thread that walks a few sequences; they do not keep up with a gather instruction whose 16 lanes walk 16 different streams.
- **Pages:** the "VRAM" uses 2 MiB pages (*transparent huge pages*). The system only allocates and zeroes a page at the first access (*page fault*), which costs tens to hundreds of microseconds per 2 MiB page if it happens in the middle of a kernel. See [18](iteration-2/18-cudamalloc-pages.md).

## Registers, pressure and spill

With 32 `zmm` registers, a float variable in a variant of W lanes occupies W/16 of them: 2 with 32 lanes, 16 with 256 lanes. When the register allocator cannot keep all live values, it does a **spill**: it writes values to the stack and rereads them later. The stack sits in L1, so each spill costs a load/store pair of a few cycles, but they occupy the memory ports and multiply inside loops. Wide variants trade fewer context switches at barriers for more register pressure; it is one of the choices that the runtime measures (see [03](iteration-2/03-automatic-width-selection.md)).

Under the x86-64 Linux calling convention (System V), **all** vector registers are caller-saved. A function call in the middle of a vectorized loop forces the spill of every live vector value, which motivated generating the cooperative barrier inline (see [15](iteration-2/15-smt-cooperative-variant.md)).

## SMT on Zen 4

Each core has two hardware threads (SMT). They share the execution units, the L1 and the L2. For code that spends time waiting for memory, the second thread hides the latency of the first; for code limited by the floating-point units, the two share the same units and each runs at about half speed. A barrier microbenchmark showed ~15 ns to synchronize two sibling threads, ~36 ns between cores of the same CCD and ~130 ns between CCDs.

## The runtime and the executor

- **Executor:** one *worker* (system thread) per reserved CPU, pinned to it (`pthread_setaffinity_np`), 28 in total. Each launch becomes a *job*; the workers take blocks and call the generated function for each group of lanes. The code is in `src/device/executor.cpp`.
- **Driver and backends:** with the *daemon* backend, the kernel runs in the `cpusimd` process, which holds the "VRAM"; with the *in-process* backend, it runs in the program itself. The interface is the same (`src/device/device.h`).
- **Fatbin and JIT:** `cucc` embeds in the executable the generic machine code and the per-thread LLVM **bitcode** (the role that PTX has in CUDA). At the first launch of each block shape, the runtime recompiles the bitcode knowing the `blockDim`, and the result is cached on disk. See [01](iteration-1/01-specialized-jit.md).

## LLVM essentials

- **LLVM IR:** the intermediate representation on which `cpusim-devc` works. It is in **SSA** form: each value is defined once, and where control paths join, a **phi** chooses the value according to the incoming path.
- **GEP** (`getelementptr`): the address arithmetic instruction (base + indices scaled by the element size). Address decomposition looks through GEPs (see [05](iteration-1/05-address-decomposition.md)).
- **Overflow flags:** `nsw` and `nuw` (*no signed/unsigned wrap*) state that an addition or multiplication does not overflow; `or disjoint` states that the operand bits do not overlap (the `or` is an addition). These flags allow distributing extensions from 32 to 64 bits (`sext`, `zext`) over additions, something that several analyses need (see [13](iteration-2/13-no-overflow-proof.md)).
- **Passes:** transformations over the IR. They appear in the documents: `instcombine` (algebraic simplification), `SROA` (promotes local variables from memory to registers), `LICM` (hoists invariant code out of loops), `GVN` (eliminates redundancies), *loop unswitching* (duplicates a loop for each value of an invariant condition), *loop unrolling*.
- **ScalarEvolution (SCEV):** the analysis that describes how a value evolves in a loop, for example `{start, +, step}` for "starts at start and adds step on each iteration". The basis of automatic prefetch (see [19](iteration-2/19-automatic-prefetch.md)).
- **Backend:** after the optimizer, the *SelectionDAG* turns the IR into x86 instructions. Vectors of "illegal" types (for example `<256 x float>`) are split into real registers in this phase.

## How we measure

- **Cycles in addition to time:** the laptop's clock varies with temperature, so many measurements are in billions of cycles (Gcyc), read from a `perf_event_open` counter that sums all threads of the process.
- **Variation between runs:** the same configuration varied by up to ±30% between runs. For that reason the comparisons use the best or the median of several runs, and variants are compared alternately within the same time interval.
- **Validation:** the benchmarks are compiled with `-fmad=false`, and the CPU result is compared bit for bit with the GPU's. The 9 HeCBench applications use the suite's own verification.

## Index

| # | Document |
|---|---|
| 01 | [Specialized JIT](iteration-1/01-specialized-jit.md) |
| 02 | [Width variants](iteration-1/02-width-variants.md) |
| 03 | [Automatic width selection](iteration-2/03-automatic-width-selection.md) |
| 04 | [Block order](iteration-1/04-block-order.md) |
| 05 | [Address decomposition](iteration-1/05-address-decomposition.md) |
| 06 | [Dynamic coalescing](iteration-1/06-dynamic-coalescing.md) |
| 07 | [Uniform loops](iteration-1/07-uniform-loops.md) |
| 08 | [If-conversion](iteration-1/08-if-conversion.md) |
| 09 | [State at barriers](iteration-1/09-state-at-barriers.md) |
| 10 | [Barrier pass](iteration-2/10-barrier-pass.md) |
| 11 | [LLVM tuning](iteration-1/11-llvm-tuning.md) |
| 12 | [Static access analysis](iteration-2/12-static-access-analysis.md) |
| 13 | [No-overflow proof](iteration-2/13-no-overflow-proof.md) |
| 14 | [Magic-number division](iteration-2/14-magic-number-division.md) |
| 15 | [SMT cooperative variant](iteration-2/15-smt-cooperative-variant.md) |
| 16 | [Splitting blocks into warps](iteration-2/16-splitting-blocks-into-warps.md) |
| 17 | [Block distribution](iteration-2/17-block-distribution.md) |
| 18 | [Pages in cudaMalloc](iteration-2/18-cudamalloc-pages.md) |
| 19 | [Automatic prefetch](iteration-2/19-automatic-prefetch.md) |
| 20 | [Concurrent kernels](iteration-2/20-concurrent-kernels.md) |
| 21 | [Compile time](iteration-2/21-compile-time.md) |
| 22 | [What did not work](iteration-2/22-what-did-not-work.md) |
