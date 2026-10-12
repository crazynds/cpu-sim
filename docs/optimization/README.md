# CudaSimulator optimizations

This folder explains, one per file, each optimization that the compiler (`cucc` and `cpusim-devc`), the runtime (`libcpusim_cudart`) and the driver's executor (`cpusimd`) received in order to run CUDA programs well on the CPU. Each document describes the problem that existed, the idea behind the solution, how it works step by step, the code excerpts that implement it, when it applies and the numbers measured before and after.

[00-basic-concepts.md](00-basic-concepts.md) gathers the mechanisms that the other documents assume: the CUDA execution model and how it becomes SIMD on the CPU, AVX-512 and its mask registers, the Zen 4 memory hierarchy (CCDs, prefetchers, huge pages), SMT and the parts of LLVM used in the optimizations.

## The machine and how we measure

- **CPU:** AMD Ryzen 9 8940HX (Zen 4, 16 cores, 2 threads per core, AVX-512, 2 x 32 MB of L3). The "device" uses 14 cores (28 workers, CPUs 2-15 and 18-31); 2 cores are left for the program.
- **Reference GPU:** NVIDIA RTX 5060 Laptop.
- **Benchmarks:** `bench/run.sh` (sample kernels, with the result checked bit for bit against the GPU) and 9 HeCBench applications compared in three forms: GPU (`nvcc`), CPU with the suite's OpenMP version, and CPU with the CUDA code on our compiler.
- **Cycles in addition to time:** the laptop changes its clock with temperature, so many measurements are in billions of cycles (Gcyc). Fine comparisons were made by alternating the variants within the same time interval, and using the median of several runs.

## The iterations

The optimizations were done in rounds. Each round has a folder with its own index:

| Iteration | What it was | Documents |
|---|---|---|
| [Iteration 1](iteration-1/README.md) | Making it viable to run CUDA on the CPU: JIT, widths, memory accesses, control flow and barriers. Includes the original record, [OPTIMIZATIONS.md](iteration-1/OPTIMIZATIONS.md) | 01, 02, 04 to 09, 11 |
| [Iteration 2](iteration-2/README.md) | Making the CUDA code on the CPU beat the HeCBench OpenMP version and making the width choice reliable | 03, 10, 12 to 22 |
| [Iteration 3](iteration-3/README.md) | The same bits in all execution modes, 16-bit types and tensor cores, the adjustments that this work exposed and a kernel written for the CPU near peak | 23 to 30 |

## The documents

The numbering is unique across all iterations, so that references do not change.

| # | Iteration | Document | In one line |
|---|---|---|---|
| 00 | | [Basic concepts](00-basic-concepts.md) | The vocabulary needed to understand the rest |
| 01 | 1 | [Specialized JIT](iteration-1/01-specialized-jit.md) | Recompiling the kernel at launch, knowing the block size |
| 02 | 1 | [Width variants](iteration-1/02-width-variants.md) | Executing 32, 64, 128... threads per call, with and without suspending at barriers |
| 03 | 2 | [Automatic width selection](iteration-2/03-automatic-width-selection.md) | Measuring the variants and keeping the best one, reliably |
| 04 | 1 | [Block order](iteration-1/04-block-order.md) | Walking the grid in bands so that simultaneous blocks share cache |
| 05 | 1 | [Address decomposition](iteration-1/05-address-decomposition.md) | Separating the common part of the address from the part that changes per thread |
| 06 | 1 | [Dynamic coalescing](iteration-1/06-dynamic-coalescing.md) | Replacing gathers with contiguous loads when the pattern allows |
| 07 | 1 | [Uniform loops](iteration-1/07-uniform-loops.md) | Loops that all threads execute identically do not need masks |
| 08 | 1 | [If-conversion](iteration-1/08-if-conversion.md) | Turning small `if`s into selects, without a branch |
| 09 | 1 | [State at barriers](iteration-1/09-state-at-barriers.md) | Saving less state when a warp stops at a `__syncthreads` |
| 10 | 2 | [Barrier pass](iteration-2/10-barrier-pass.md) | The last warp to arrive crosses the barrier without stopping |
| 11 | 1 | [LLVM tuning](iteration-1/11-llvm-tuning.md) | Extra optimizer options and passes |
| 12 | 2 | [Static access analysis](iteration-2/12-static-access-analysis.md) | Proving at compile time that an access is contiguous |
| 13 | 2 | [No-overflow proof](iteration-2/13-no-overflow-proof.md) | Using the CUDA limits to prove that indices do not overflow |
| 14 | 2 | [Magic-number division](iteration-2/14-magic-number-division.md) | Division by a common value becomes multiplication |
| 15 | 2 | [SMT cooperative variant](iteration-2/15-smt-cooperative-variant.md) | The two threads of a core share one block |
| 16 | 2 | [Splitting blocks into warps](iteration-2/16-splitting-blocks-into-warps.md) | With few blocks, distributing warps instead of blocks |
| 17 | 2 | [Block distribution](iteration-2/17-block-distribution.md) | Each block always on the same core, all advancing together |
| 18 | 2 | [Pages in cudaMalloc](iteration-2/18-cudamalloc-pages.md) | Actually allocating the memory at the time of `cudaMalloc` |
| 19 | 2 | [Automatic prefetch](iteration-2/19-automatic-prefetch.md) | Requesting data ahead of time, where the hardware cannot |
| 20 | 2 | [Concurrent kernels](iteration-2/20-concurrent-kernels.md) | Real streams and several kernels at the same time |
| 21 | 2 | [Compile time](iteration-2/21-compile-time.md) | Compiling and testing faster (and working around an LLVM bug) |
| 22 | 2 | [What did not work](iteration-2/22-what-did-not-work.md) | Ideas that were tested and did not help, and why |
| 23 | 3 | [Reproducibility and stability](iteration-3/23-reproducibility-and-stability.md) | The same bits in all execution modes, and FMA contraction decided in the front end |
| 24 | 3 | [16-bit types](iteration-3/24-16-bit-types.md) | `__half` and `__nv_bfloat16` computed in `float` and rounded only once |
| 25 | 3 | [Tensor cores](iteration-3/25-tensor-cores.md) | `wmma::mma_sync` becomes one call per warp to `vdpbf16ps`, `vpdpbusd` or F16C with FMA |
| 26 | 3 | [Compiler and runtime changes](iteration-3/26-compiler-and-runtime-changes.md) | Division by a constant in divergent code, the generic-variant budget and the JIT cache key |
| 27 | 3 | [A kernel written for the CPU, near peak](iteration-3/27-kernel-at-peak.md) | GEMM in CUDA for the `cucc` model: 94% (fp32) and 93% (bf16) of one core's peak, with the bfloat16 `mma_sync` inline and the accumulators in registers |
| 28 | 3 | [Function calls that are not inlined](iteration-3/28-function-calls.md) | Recursion, function pointers and virtual calls: a W-lane version of the function, called with the warp's mask |
| 29 | 3 | [The stress suite](iteration-3/29-stress-suite.md) | 631 cases checked against the GPU: from 174 passing to 629, what broke and why, and the slow cases left for later |
| 30 | 3 | [Launches and the slow cases](iteration-3/30-launches-and-hot-spots.md) | An empty kernel from 42 us to 1.7 us; atomics merged per warp; against OpenMP, 335 of 541 cases faster (233 before) |

## Suggested reading order

1. **Foundation:** 00, 01, 02 and 03. They explain how a CUDA kernel becomes CPU code and how the runtime chooses the way to execute it.
2. **Memory accesses in the generated code:** 05, 06, 12, 13 and 14.
3. **Control flow and barriers:** 07, 08, 09, 10 and 15.
4. **How the work is distributed among the cores:** 04, 16, 17, 18, 19 and 20.
5. **Behind the scenes:** 11, 21 and 22.
6. **Correctness and new features:** 23, 24, 25 and 26. Bit-for-bit reproducibility, 16-bit types and tensor cores.
7. **Writing the kernel for the CPU:** 27. A GEMM near the theoretical peak, and what the compiler needed to change for it.

The shorter original record of iteration 1 is in [iteration-1/OPTIMIZATIONS.md](iteration-1/OPTIMIZATIONS.md).
