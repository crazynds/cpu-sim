# 5. Tensor cores and the other benchmarks

## Tensor cores: capability check

`nvcuda::wmma` runs on the CPU's dot-product instructions (table in tutorial 1). As on a GPU, the device's compute capability says which input types exist, and a kernel using a type the device lacks fails to launch with `cudaErrorNoKernelImageForDevice`. Details: [docs/optimization/iteration-3/25-tensor-cores.md](../optimization/iteration-3/25-tensor-cores.md).

```sh
build/cpu-smi -q | grep -iE 'compute|tensor'
```

`CPUSIM_TENSOR` can only lower the level, to simulate an older device. It is read by the device, that is, by the driver: start one with it, and the kernels of the missing types must be refused:

```sh
build/cucc -O3 -std=c++20 -I samples/common samples/tensor_gemm.cu -o build/bench/tensor_gemm
build/bench/tensor_gemm all 1024 2                                  # every type the CPU supports
build/cpusimd --stop && CPUSIM_TENSOR=int8 build/cpusimd --daemon
build/bench/tensor_gemm all 1024 2                                  # compute capability 7.5: no bf16
build/cpusimd --stop && CPUSIM_TENSOR=fp16 build/cpusimd --daemon
build/bench/tensor_gemm all 1024 2                                  # compute capability 7.0: fp16 only
build/cpusimd --stop && build/cpusimd --daemon                      # back to the full level
```

The sample checks the capability first, like a CUDA program would, and skips what is missing. The tests `cuda_wmma`, `cuda_wmma_shapes` and `cuda_wmma_subbyte` run with both lowered levels and check that the launch itself is refused.

## Every tensor core operation: `bench/tensor_ops.cu`

One program measures each operation: a GEMM of n x n through WMMA for every input type and shape (fp16 to fp32 and to fp16, bf16, tf32, fp64, int8, uint8, int4, uint4, and the 1-bit xor and and with popcount), one warp per tile of C. Built with `nvcc` it runs on the GPU, built with `cucc` on the simulator; `omp` runs, for each type, a C++ function with OpenMP that computes the same product from the same inputs with the same semantics, on the CPU without CUDA:

```sh
build/cucc -O3 -std=c++20 -fopenmp -Xcompiler -march=native bench/tensor_ops.cu -o build/bench/tensor_ops_cpu
nvcc -O3 -std=c++20 -arch=native -Xcompiler -fopenmp,-march=native bench/tensor_ops.cu -o build/bench/tensor_ops_gpu
build/bench/tensor_ops_gpu cuda      # the GPU's tensor cores
build/bench/tensor_ops_cpu cuda      # the simulator (through the driver)
build/bench/tensor_ops_cpu omp       # plain C++ with OpenMP
build/bench/tensor_ops_cpu cuda 4096 # another size (a multiple of 128)
```

Each line gives the operation, the tile shape, the time of one GEMM (best of 3 after a warm-up), the rate in T(FL)OPS (2n^3 operations), and a check against an exact reference on 16 rows. Types the device's compute capability lacks are skipped. Compare the three runs per operation; compare also `cuda` with `./peak` (tutorial 2) for the types it covers: this GEMM loads every fragment from memory, so it measures the tensor path together with the loads that feed it.

## Tensor core throughput

`samples/tensor_gemm.cu` is a GEMM written for a GPU (blocks of 4 warps, shared-memory staging), compiled unmodified:

```sh
build/bench/tensor_gemm bf16
nvcc -O3 -std=c++20 -arch=native -I samples/common samples/tensor_gemm.cu -o build/bench/tensor_gemm_gpu && build/bench/tensor_gemm_gpu bf16
```

`samples/cpu_gemm.cu` is the same computation written for the simulator's execution model (one warp per block, the accumulator fragment kept in registers through the loop over k, A loaded one step ahead):

```sh
build/cucc -O3 -std=c++20 -I samples/common samples/cpu_gemm.cu -o build/bench/cpu_gemm
build/bench/cpu_gemm bf16
build/bench/cpu_gemm fp32
nvcc -O3 -std=c++20 -arch=native -I samples/common samples/cpu_gemm.cu -o build/bench/cpu_gemm_gpu && build/bench/cpu_gemm_gpu
```

What to look for:

- Compare `cpu_gemm bf16` with `build/bench/peak bf16` (tutorial 2) run right before it: the distance is what the loads and stores of a real GEMM cost against a loop that only multiplies.
- Compare `tensor_gemm` and `cpu_gemm` on both devices. On the CPU, the GPU-style kernel pays for shared-memory staging and barriers, which the CPU-style one avoids; on the GPU, see which structure the hardware prefers per type (the fp32 path of `cpu_gemm` uses no tensor cores, so it is not a GPU fp32 reference).
- The fp32 path of `cpu_gemm` is the "before" kernel of tutorial 4 (12×32 per warp); the duel uses the newer one.

## The kernel comparison of `bench/run.sh`

[BENCHMARKS.md](../../BENCHMARKS.md) compares unmodified GPU kernels (matmul, mandelbrot, heat2d, transpose, scan, bitonic sort) on the GPU, on `cucc`, and as plain C++ with OpenMP. Every CPU result is checked bit for bit against the GPU's (a hash of the output):

```sh
bench/run.sh gpu       # needs nvcc
bench/run.sh cpu       # prints OK/DIFF per output against the GPU run
bench/run.sh native    # plain C++, 1 thread and NATIVE_THREADS threads (default 28)
```

`OK` lines are the equality checks; `DIFF` means a mismatch and should never appear. These kernels are compiled with `-fmad=false` on both sides, so that the float results are comparable bit for bit. Set `NATIVE_THREADS` to the number of logical CPUs of your device (`cpu-smi -q`) so the C++ version gets the same hardware.

## Samples at full size

The samples in `samples/` run at GPU-sized workloads by default and print `PASS`/`FAIL` against a CPU reference:

```sh
build/cucc -O3 -std=c++20 -fmad=false -I samples/common samples/nbody.cu -o build/bench/nbody
build/bench/nbody
```

The references are single-threaded and naive, so some samples take a minute or two. `ctest -L samples` runs them at test size (`-DSAMPLE_TEST`).
