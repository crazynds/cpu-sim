# 2. Arithmetic peak and clock

Before judging any compute-bound kernel, measure what the device can do at all. [bench/peak.cu](../../bench/peak.cu) is a CUDA program whose kernels do nothing but independent multiply-adds; it runs on the device's multiprocessors like any other kernel, so it measures the same CPUs, at the same clock, that a benchmark gets.

## The theory

Each multiprocessor of the simulator is one physical core. Per cycle, a core with AVX-512 can retire a number of 512-bit FMAs that depends on the microarchitecture (one on cores that split 512-bit operations into two 256-bit halves, two on cores with two 512-bit FMA units). Each 512-bit fp32 FMA is 16 lanes × 2 FLOPs = 32 FLOPs; each 512-bit `vdpbf16ps` is 16 lanes × 2 pairs × 2 = 64 FLOPs. The SMT siblings of a core share its units. So:

    peak (GFLOPS) = multiprocessors × FMA units per core × FLOPs per instruction × clock (GHz)

The clock is the unknown: under an all-core AVX-512 load a CPU holds less than its single-core boost, and how much less depends on its power limits and cooling. The program measures the result directly.

## Build and run

```sh
build/cucc -O3 -std=c++20 bench/peak.cu -o peak
./peak            # fp32 and bf16, about 2 s each
./peak fp32 5     # only fp32, 5 s
```

with `cpu-smi` in another terminal (tutorial 1) to see the clock the device holds during the run:

```sh
build/cpu-smi --query-gpu=temperature.gpu,clocks.sm,utilization.gpu --format=csv -l 0.5
```

The output gives the device, its compute capability and multiprocessors, and:

- `fp32 FMA`: each thread runs 12 independent `fmaf` chains. A warp is two 512-bit registers per value, so each worker has 24 independent FMA chains, more than the FMA latency requires: this is the FMA throughput of the device.
- `bf16 tensor`: each warp runs `wmma::mma_sync` on a 16×16×16 tile with the accumulator in registers. It needs compute capability 8.0 (AVX512-BF16, see the table in tutorial 1) and is skipped otherwise. Each `mma_sync` is 128 `vdpbf16ps` per warp plus the loads of the A operand, so this is the ceiling of tensor code through WMMA, a little below the raw instruction peak.

The same source builds for a GPU, for the same measurement there:

```sh
nvcc -O3 -std=c++20 -arch=native bench/peak.cu -o peak_gpu && ./peak_gpu
```

## Things to check

- **The clock**: divide the `fp32 FMA` result by (multiprocessors × 32 FLOPs × FMA units per core) and compare with `clocks.sm` from `cpu-smi`. If they agree, the kernel retires FMAs at the hardware rate and the number is the true ceiling.
- **Scaling with cores**: set `cores = 1` (then 2, 4, ...) in `~/.config/cpusim/device.toml`, restart the driver and run again. The device always uses both SMT threads of each core it takes, so the result is per physical core; it should grow linearly until power limits lower the clock.
- **Thermal drift**: run `./peak fp32 30` and watch `clocks.sm`. The difference between the first seconds and the end is how much your cooling costs a long benchmark.

## Using the peak as a reference

A GEMM is compute bound, so its ceiling is this peak at the clock it holds while running. Two ways to compare:

1. **Same thermal state**: run `./peak fp32`, then the GEMM right away, and divide the GEMM's GFLOPS by the peak.
2. **Same clock**: a GEMM usually holds a lower clock than this kernel, because it also moves data (loads, cache traffic, DRAM), which costs power. Note `clocks.sm` during the GEMM, compute the peak at that clock with the formula above, and divide. If the GEMM is close to the peak at its own clock, the code wastes few cycles and what is left is power, not code; that is why tutorial 4 looks for fewer memory operations per FMA and less memory traffic.

`clocks.sm` comes from cpufreq (`scaling_cur_freq`), an estimate the kernel driver refreshes every few milliseconds: use it for trends and ratios, not for exact efficiencies.
