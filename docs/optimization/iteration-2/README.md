# Iteration 2

The second round: it started by investigating why the tiled matmul did not beat the naive one, went through the comparison with 9 HeCBench applications (GPU, OpenMP and our compiler) and ended with the CUDA code on the CPU faster than the suite's own OpenMP version in 5 of the 7 applications where the comparison is valid, and tied in the other 2. It also made the automatic width selection reliable and validated.

| # | Document | In one line |
|---|---|---|
| 03 | [Automatic width selection](03-automatic-width-selection.md) | Measuring the variants and keeping the best one, reliably |
| 10 | [Barrier pass](10-barrier-pass.md) | The last warp to arrive crosses the barrier without stopping |
| 12 | [Static access analysis](12-static-access-analysis.md) | Proving at compile time that an access is contiguous |
| 13 | [No-overflow proof](13-no-overflow-proof.md) | Using the CUDA limits to prove that indices do not overflow |
| 14 | [Magic-number division](14-magic-number-division.md) | Division by a common value becomes multiplication |
| 15 | [SMT cooperative variant](15-smt-cooperative-variant.md) | The two threads of a core share one block |
| 16 | [Splitting blocks into warps](16-splitting-blocks-into-warps.md) | With few blocks, distributing warps instead of blocks |
| 17 | [Block distribution](17-block-distribution.md) | Each block always on the same core, all advancing together |
| 18 | [Pages in cudaMalloc](18-cudamalloc-pages.md) | Actually allocating the memory at the time of `cudaMalloc` |
| 19 | [Automatic prefetch](19-automatic-prefetch.md) | Requesting data ahead of time, where the hardware cannot |
| 20 | [Concurrent kernels](20-concurrent-kernels.md) | Real streams and several kernels at the same time |
| 21 | [Compile time](21-compile-time.md) | Compiling and testing faster (and working around an LLVM bug) |
| 22 | [What did not work](22-what-did-not-work.md) | Ideas that were tested and did not help, and why |

This iteration also added parts to three documents of iteration 1: [06](../iteration-1/06-dynamic-coalescing.md), [09](../iteration-1/09-state-at-barriers.md) and [11](../iteration-1/11-llvm-tuning.md).

## Results of the 9 HeCBench applications at the end of the iteration

Best of 3 runs; GPU = RTX 5060 with `nvcc`; OMP = the suite's OpenMP version on 32 CPUs; cucc = original CUDA code on our compiler, on 28 CPUs.

| Application | Metric | GPU | OMP | cucc | cucc vs OMP |
|---|---|---|---|---|---|
| nbody | GFLOP/s | 2763 | 135 | 876 | 6.5x faster |
| mandelbrot | ms per kernel | 0.184 | 13.5 | 1.69 | 8.0x faster |
| black-scholes | ms per kernel | 5.71 | 127 | 50.3 | 2.5x faster |
| iso2dfd | ms per kernel | 0.181 | 0.581 | 0.386 | 1.5x faster |
| heat2d | ms per iteration | 0.454 | 9.23 | 4.16 | 2.2x faster |
| babelstream | GB/s (Triad) | 330 | 33.4 | 32.1 | tie (RAM bandwidth) |
| matrix-rotate | ms per kernel | 6.48 | 4.75 | 4.73 | tie (3.97 against 4.21 ms in steady state) |
| fdtd3d | ms per kernel | 0.264 | wrong result | 2.49 | no valid comparison |
| triad | GB/s (with copies) | 46.1 | not comparable | 41.2 | no valid comparison |

Previous: [iteration 1](../iteration-1/README.md). Next: [iteration 3](../iteration-3/README.md). Back to the [index](../README.md).
