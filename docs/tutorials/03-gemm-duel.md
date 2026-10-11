# 3. The fp32 GEMM duel

C = A·B in fp32, square n×n, with three contenders, each written the best way for its own target:

| Contender | Built with | What runs |
|---|---|---|
| GPU | `nvcc` | cuBLAS `cublasGemmEx` with `CUBLAS_COMPUTE_32F_FAST_TF32` (tensor cores), `cudaMalloc` + copies over PCIe |
| `cucc` | `build/cucc` | the same source's CUDA path, written for the simulator: `cudaMallocManaged`, packing kernels, a 6×64 micro-kernel per block of 64 threads |
| OpenMP | `build/cucc` (host code) | BLIS-style C++ with AVX-512 vector types: packing, KC/MC/NC cache blocks, 8×32 micro-kernel |

All three are in one file, [bench/gemm_fp32.cu](../../bench/gemm_fp32.cu). Read it before running: the claims below are about that code.

## What is measured

For each n (a multiple of 1536):

- **kernel (ms)**: mean time of the multiplication alone, over 20 repetitions (5 for the large sizes). For `cucc` and the GPU this is the GEMM kernel only; for OpenMP it is the whole call, which includes its packing.
- **total (ms)**: one multiplication from data in host memory to the result in host memory. GPU: two uploads, the kernel, one download. `cucc`: the two packing kernels and the GEMM kernel; no copies, because the program keeps A, B and C in managed memory, which on the simulator is host memory. OpenMP: the same as its kernel time.
- **GFLOPS**: 2n³ / kernel time.
- **check**: 32 rows of C compared exactly with a double-precision reference. The inputs are small integers in [-4, 4], so every product and every partial sum is an exact integer in fp32 (and in TF32, whose 10-bit mantissa holds these values): any error, of any size, prints `FAIL`.

About fairness: in the `cucc` total, filling A and B is not timed, because a program written for the simulator would produce its data directly in managed memory, as the OpenMP program produces it in its arrays. The GPU total does count the copies, because on a discrete GPU they cannot be avoided.

The `cucc` version adapts to the device through `cudaGetDeviceProperties`, as CUDA code adapts to a GPU: the number of rows per block comes from `multiProcessorCount` (enough blocks to keep every multiprocessor busy), and the packing uses streaming stores only when the packed copies do not fit in `l2CacheSize` (on the simulator, the CPU's last-level cache).

## Run it

With the driver running (tutorial 1), the script builds both binaries and runs everything:

```sh
bench/gemm_duel.sh                    # 1536 3072 4608 6144, 2 pairs per size, 25 s pauses
PAIRS=1 PAUSE=15 bench/gemm_duel.sh 3072 6144     # quicker
```

It prints the GPU table first (if `nvcc` is available), then one line per CPU run, then the mean per size:

```
== mean per size (cuda = cucc)
     n   cucc kernel   cucc total     omp total    kernel x    total x
```

where `kernel x` = OpenMP time / `cucc` kernel time and `total x` = OpenMP time / `cucc` total time (above 1 means `cucc` is faster).

To run by hand:

```sh
build/cucc -O3 -std=c++20 -fopenmp bench/gemm_fp32.cu -o build/bench/gemm_cpu
nvcc -O3 -std=c++20 -arch=native -Xcompiler -fopenmp bench/gemm_fp32.cu -lcublas -o build/bench/gemm_gpu
build/bench/gemm_gpu cuda
build/bench/gemm_cpu cuda 1536 3072
build/bench/gemm_cpu omp 1536 3072
```

`build/bench/gemm_cpu omp` uses every hardware thread by default; `OMP_NUM_THREADS` changes it. To give OpenMP exactly the device's CPUs, take the list from `cpu-smi -q` (`Logical CPUs`) and pin the threads to it; for a device on CPUs 4-7 and 12-15, for example: `OMP_PLACES="{4}:4:1,{12}:4:1" OMP_PROC_BIND=close OMP_NUM_THREADS=8`.

## Reading the results

- **check must be `ok`** in every line. A `FAIL` invalidates the line.
- **A result is real if the ratio keeps its side of 1 across pairs.** Single numbers move with the clock (tutorial 1); the ratio within a pair is what to compare.
- **Compare the kernel GFLOPS with `build/bench/peak fp32`** (tutorial 2) run right before: that is how far each CPU contender is from the hardware.
- **Kernel versus total**: the gap between the two `cucc` columns is the packing; for OpenMP the packing is inside its single number. For the GPU the gap is mostly PCIe copies.

## Reproduce the "before"

The first version of the duel used the same kernel as `samples/cpu_gemm.cu` (12×32 per warp), `cudaMalloc` with copies, and packing with one thread per element. It is in git, so the two versions can run side by side:

```sh
git show ae2890d:bench/gemm_fp32.cu > /tmp/gemm_fp32_before.cu
build/cucc -O3 -std=c++20 -fopenmp /tmp/gemm_fp32_before.cu -o build/bench/gemm_before
build/bench/gemm_before cuda 6144; sleep 30
build/bench/gemm_cpu cuda 6144
```

Tutorial 4 breaks the difference down step by step.

## Things to check yourself

- **The check catches errors**: in `sgemm`, change `acc[r] = pc == 0 ? 0.0f : c[(long)r * N]` to start from `0.0f` always (dropping every pass of k but the last); the `cucc` run must print `FAIL`. Undo it afterwards.
- **The `cucc` version is plain CUDA**: it compiles with `nvcc` too (the GPU build compiles it, it just does not run it). Nothing in it calls simulator-specific APIs; `cudaMallocManaged`, `__stcs` and `cudaGetDeviceProperties` are standard CUDA.
- **The OpenMP version is not a strawman**: it is a BLIS-structured GEMM with an AVX-512 micro-kernel. It is not OpenBLAS or MKL, which would be faster; an optimized BLAS is a different comparison.
