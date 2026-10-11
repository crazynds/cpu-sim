# 5. Tensor cores and the other benchmarks

## Tensor cores: capability check

`nvcuda::wmma` runs on the CPU's dot-product instructions (table in tutorial 1). As on a GPU, the device's compute capability says which input types exist, and a kernel using a type the device lacks fails to launch with `cudaErrorNoKernelImageForDevice`. Details: [docs/optimization/iteration-3/25-tensor-cores.md](../optimization/iteration-3/25-tensor-cores.md).

```sh
build/cpu-smi -q | grep -iE 'compute|tensor'
```

`CPUSIM_TENSOR` can only lower the level, to simulate an older device. With it, the kernels of the missing types must be refused:

```sh
build/cucc -O3 -std=c++20 -I samples/common samples/tensor_gemm.cu -o tensor_gemm
CPUSIM_BACKEND=inprocess ./tensor_gemm all 1024 2                    # every type the CPU supports
CPUSIM_TENSOR=int8 CPUSIM_BACKEND=inprocess ./tensor_gemm all 1024 2 # compute capability 7.5: no bf16
CPUSIM_TENSOR=fp16 CPUSIM_BACKEND=inprocess ./tensor_gemm all 1024 2 # compute capability 7.0: fp16 only
```

(`CPUSIM_TENSOR` is read by the device, so these runs use the in-process backend; a running driver keeps its own level.) The sample checks the capability first, like a CUDA program would, and skips what is missing. The test `cuda_wmma` runs with both lowered levels and checks that the launch itself is refused.

## Tensor core throughput

`samples/tensor_gemm.cu` is a GEMM written for a GPU (blocks of 4 warps, shared-memory staging), compiled unmodified:

```sh
./tensor_gemm bf16
nvcc -O3 -std=c++20 -arch=native -I samples/common samples/tensor_gemm.cu -o tensor_gemm_gpu && ./tensor_gemm_gpu bf16
```

`samples/cpu_gemm.cu` is the same computation written for the simulator's execution model (one warp per block, the accumulator fragment kept in registers through the loop over k, A loaded one step ahead):

```sh
build/cucc -O3 -std=c++20 -I samples/common samples/cpu_gemm.cu -o cpu_gemm
./cpu_gemm bf16
./cpu_gemm fp32
nvcc -O3 -std=c++20 -arch=native -I samples/common samples/cpu_gemm.cu -o cpu_gemm_gpu && ./cpu_gemm_gpu
```

What to look for:

- Compare `cpu_gemm bf16` with `./peak bf16` (tutorial 2) run right before it: the distance is what the loads and stores of a real GEMM cost against a loop that only multiplies.
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
build/cucc -O3 -std=c++20 -fmad=false -I samples/common samples/nbody.cu -o nbody
./nbody
```

The references are single-threaded and naive, so some samples take a minute or two. `ctest -L samples` runs them at test size (`-DSAMPLE_TEST`).
