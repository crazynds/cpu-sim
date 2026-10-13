# 31. Micro-optimizations, and gathers and scatters with 32-bit indices

> **In one sentence:** a study of small code-generation changes (instruction order, interleaving the groups' memory and arithmetic, masks on loop-carried values) found that the hardware already does them: with both SMT threads busy, the compute kernels are bound by the FP units' throughput and the rest by memory, except where gathers and scatters dominate; there, two changes paid off: addresses used in another basic block are recomputed instead of kept as 64-bit pointers per lane, and the gathers and scatters that remain use 32-bit indices (bitonic sort 1.18x, 31 of 104 cases at least 5% faster).

The general concepts (lanes, variants, address decomposition) are in [00-basic-concepts.md](../00-basic-concepts.md); the access patterns tested before a gather are in [06](../iteration-1/06-dynamic-coalescing.md).

## Where the cycles go

Zen 4's top-down counters (`perf stat -M PipelineL2`, attached to the driver) over the six `bench/` kernels:

| Kernel | IPC | Main limit |
|---|---|---|
| matmul | 1.03 | FP units 34%, memory 19% (56% of the samples on the load of B) |
| mandelbrot | 0.94 | FP units 52% |
| transpose | 0.21 | memory 70% |
| heat2d | 0.35 | memory 57% |
| scan | 0.69 | microcoded operations 14% of the retired ones |
| bitonic_sort | 0.76 | microcoded operations 22%; about 45% of the cycles on gathers and scatters |

Mandelbrot reaches 1.24 TFLOP/s with 94% of its lanes active, about 70% of the 14 cores' peak without FMA. Little is left in it.

## What did not help

- **Instruction order.** The kernels rebuilt with LLVM's other machine schedulers (`-misched=ilpmax`, `-misched-topdown`, `-enable-post-misched`, through `CPUSIM_LLVM_FLAGS`) and measured in alternating pairs: bitonic and heat2d within ±3%, mandelbrot 9 to 17% slower. Zen 4 reorders a 320-entry window by itself.
- **Interleaving the groups' memory and arithmetic.** It already happens twice: the 64- and 128-lane variants put several warps in one instruction stream (the width selection picks them where they win), and the two SMT threads of a core run two independent blocks.
- **Masks off the loop-carried values.** Mandelbrot's loop updates `x` and `y` under the loop mask (`vaddps ... {k2}`), which puts the mask's compare and `kand` chain on the critical path, although `x` and `y` are dead after the loop. A hand-written copy without those masks: 7% faster alone on a core (7.74 to 7.22 cycles per trip), 0% with the SMT sibling busy (13.3 both ways), where the FP units are the limit, not latency.
- **32-bit indices for their own sake.** Measured on L1-resident data, the hardware's cost is per element:

| Operation (16 lanes) | Cycles |
|---|---|
| `vpgatherdd` (32-bit indices, 1 instruction) | 10.2 |
| `vpgatherqd` (64-bit indices, 2 instructions) | 11.5 |
| `vpscatterdd` | 11.7 |
| `vpscatterqd` (2 instructions) | 11.9 |

## What did

### Addresses recomputed where they are used

In bitonic sort, `s[pos]` is computed before an `if` and stored to inside it. A value used in another basic block gets a slot (a stack variable that holds it between blocks), unless it is cheap to recompute there; that rematerialization took arithmetic and casts but not address computations (GEPs). So the address went to a slot as W 64-bit pointers, and the store saw an opaque pointer: no base, no contiguity test, a scatter of absolute addresses every time. `remat_ok` now accepts a varying GEP whose operands qualify, and the store sees base + offsets again (`src/compiler/simt.cpp`).

### 32-bit indices, always correct

LLVM does not narrow the indices by itself: even when it proves they fit, its gather lowering loses the uniform base and keeps 64-bit ones. The compiler now emits `vpgatherdd`/`vpscatterdd` (and `vpgatherdq`/`vpscatterdq` for 8-byte elements) directly, counting bytes from the access's root pointer: the parameter, the shared memory base, the pointer loaded from memory, before any GEP (`gather16`, `scatter16`). Counting from the root and not from the base with the uniform part already added matters: the uniform and varying parts may cancel, but an access `p[i]` must stay inside `p`'s object (C++ forbids pointer arithmetic out of it), so `|address - root|` is below that object's size. The index fits in 32 signed bits whenever the object is under 2 GiB:

- **Shared memory and per-thread arrays:** always. Their gathers and scatters are 32-bit unconditionally.
- **Global memory:** per root pointer. The allocators place every allocation of 2 GiB or more ("huge") at the top of the VRAM and of the shared pool, and each launch passes the span `[huge_lo, huge_hi]` of the huge ones in the context (`cpusim_warp_ctx`, kernel ABI 12). The generated code compares the root with it (two loads and two compares on a uniform value, hoisted out of loops with the root): outside, 32-bit indices; inside, the 64-bit form. A program with a 3 GiB buffer keeps the 32-bit form for all its other buffers. `huge_hi` is inclusive: a pointer one past the end of a huge object is valid C++ and may index back into it (`end[-1 - i]`). A small allocation that lands inside the span (between two huge ones) only loses the 32-bit form. Device `malloc` refuses objects of 2 GiB or more (`cap_device_malloc`), as a GPU does with a heap smaller than that; the runtime only sees the VRAM and shared pool. `CPUSIM_SMALL_OFFSETS=0` on the driver makes the span cover everything (the 64-bit form), for comparisons.
- **Opaque pointers** (one pointer per lane loaded from memory, with no common root): 64-bit, as before. Placing all device memory in one low 4 GiB window would cover them too, but the VRAM is larger than that by default and such accesses are rare.

Bytes and 16-bit values are gathered as the aligned 32-bit word holding them, as before, now from the root rounded down to 4 bytes.

## Results

The stress program `mem_large` checks the boundary: one kernel gathers from a 3 GiB allocation (positions beyond 2 GiB) and from a small one, another indexes back from the 3 GiB buffer's end pointer, another scatters into it, and bytes are gathered beyond 2 GiB of a 2 GiB + 1 MiB buffer. It passes on the GPU and on `cucc`, and with the 32-bit form forced for every root two of its cases fail, as they should.

Paired runs against the previous build (`bd78892`), each with its own driver, alternated over two rounds, means compared; the six `bench/` kernels and 13 stress programs (`wl_*` and `sync_atomic_global`), 104 cases. The results are unchanged bit for bit; the stress suite passes 631 of 633 cases (the two left are still `j0` and `y0`; the two added are the 2 GiB cases of `mem_large` above).

| Case | Before (ms) | After (ms) | Speedup |
|---|---|---|---|
| `dequant_int8_per_channel` | 2.16 | 1.70 | 1.27x |
| `bfs_256K_nodes` | 2.00 | 1.65 | 1.21x |
| bench `bitonic_sort` | 673 | 571 | 1.18x |
| `transpose_naive` (stress) | 9.66 | 8.21 | 1.18x |
| `softmax_rows4096` | 4.33 | 3.70 | 1.17x |
| `bias_relu_nchw` | 4.01 | 3.46 | 1.16x |
| bench `matmul_naive` | 175 | 152 | 1.15x |
| `lower_bound_4M_queries` | 11.4 | 10.0 | 1.14x |
| bench `scan_blocks` | 354 | 314 | 1.12x |
| `transpose_rect_7001x4003` | 7.77 | 6.91 | 1.12x |

Over all 104 cases: 31 at least 5% faster, 9 at least 5% slower (the worst, `fused_chain20`, 0.94x, has no gather: noise of a two-round mean), 64 within 5%; geometric mean 1.015, 1.034 over the 88 cases of 1 ms or more.

## What is left

- **A gather inside a window.** When the 16 indices fall within 32 consecutive elements (bitonic's strides below 16, a scan's first level, exchanges between neighbors), two aligned loads and a `vpermt2d` replace the gather: 7.3 cycles against 19.4 in a microbenchmark. A scatter has no such shortcut (it needs the inverse permutation).
- **Width per region.** A loop with many live values could run as two 32-lane halves inside a 64-lane variant, instead of the whole kernel choosing one width; matmul shows the cost it would avoid (128 lanes loses to 64 by spills).

Previous: [30](30-launches-and-hot-spots.md). Next: [32](32-warp-size.md). Back to the [iteration 3 index](README.md).
