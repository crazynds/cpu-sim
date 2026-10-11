# Benchmarks: GPU, our executor and plain C++ on the CPU

Measured on 2026-10-09, laptop plugged in (`balance_performance` profile), with `bench/run.sh gpu`, `cpu` and `native`. Each number is the best of several runs of the kernel only, without copies.

- **GPU**: RTX 5060 Laptop, code compiled with `nvcc -fmad=false`.
- **Our executor**: the same `.cu`, unmodified, compiled with `cucc` and running on the executor (28 threads on 14 cores, JIT per block shape, width chosen by measurement).
- **C++ 1 core / 28 threads**: plain C++ versions (`bench/native/`, `g++ -O3 -march=native -fopenmp -ffp-contract=off`), with the same number of threads the executor uses.

All results match the GPU bit for bit (output hash).

## GFLOP/s

FLOPs: matmul = 2·M·N·K (137.8 GFLOP, the useful work, the same for all versions); mandelbrot and heat_step = exact profiling count (`CPUSIM_PROFILE`) at those dimensions.

| Kernel | GFLOP | GPU | Our executor | C++ 1 core | C++ 28 threads |
|---|---|---|---|---|---|
| matmul naive (C++: vectorized ikj) | 137.8 | 868 (159 ms) | 562 (245 ms) | 13.8 (10.0 s) | 211 (654 ms) |
| matmul tiled (C++: cache blocking) | 137.8 | 1124 (123 ms) | 421 (327 ms) | 36.3 (3.8 s) | 251 (548 ms) |
| mandelbrot | 535.0 | 5745 (93 ms) | 740 (723 ms) | 4.6 (117 s) | 92 (5.8 s) |
| heat_step (1 step) | 0.34 | 184 (1.8 ms) | 20.8 (16.1 ms) | 9.2 (36.3 ms) | 24.2 (13.9 ms) |

## Kernels without floating point (time)

| Kernel | GPU | Our executor | C++ 1 core | C++ 28 threads |
|---|---|---|---|---|
| transpose naive (C++: direct) | 13.7 ms | 141 ms | 1069 ms | 247 ms |
| transpose tiled (C++: 32x32 blocks) | 11.7 ms | 136 ms | 386 ms | 171 ms |
| scan_blocks (segments of 512) | 20.7 ms | 424 ms | 176 ms | 158 ms |
| scan of the whole array | 34.2 ms | 457 ms | 168 ms | 255 ms |
| bitonic_sort (C++: `std::sort` per segment) | 25.8 ms | 702 ms | 5147 ms | 262 ms |

## Reading the results

- **Compute-bound kernels.** Our executor running unmodified CUDA is faster than hand-written multi-core C++: matmul 2.2x to 2.7x, mandelbrot 8x. The generated code uses AVX-512 with access patterns resolved by the JIT. Against the GPU, it is 1.5x (matmul naive) to 7.8x (mandelbrot) slower.
- **Memory-bound kernels** (`heat_step`, `transpose`). All CPU versions are close to each other, and the GPU has the advantage because of memory bandwidth.
- **`scan` and `bitonic`** (barriers + divergent branches) are the executor's weak point: it falls behind even the 1-core C++ on `scan`.
- The C++ is what one would write by hand without a library: an optimized BLAS (OpenBLAS) would make the matmul much faster, but it sums in a different order and would not give a result identical to the GPU's.
