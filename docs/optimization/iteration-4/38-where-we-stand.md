# 38. Where we stand: 72 programs drawn at random, against the H100 and OpenMP

> **In one sentence:** on 72 HeCBench programs drawn at random, every one that runs on the GPU without CUB or Thrust also runs and passes on `cucc` (52 of 52); its kernels are faster than the same program's OpenMP version in 15 of 27 comparable programs, tied in 3 and slower in 9, and slower than an H100 in 43 of 48, 20 of them by more than 10x; it compiles a program in 1.7 s (median), against 1.3 s for `nvcc`.

Earlier comparisons with OpenMP ([30](../iteration-3/30-launches-and-hot-spots.md), [33](../iteration-3/33-xeon-server.md)) used programs we had chosen, many of them for being slow. This one draws them at random.

## Method

| Item | Setting |
|---|---|
| Programs | 72 of the 349 HeCBench programs with both a CUDA and an OpenMP version, drawn with a fixed seed in two draws (32, then 40 more); no program modified; each run with its Makefile's `run` arguments |
| Machine | 2 x Intel Xeon Gold 6548Y+ (64 cores, 128 threads, 2 NUMA nodes), 1 TB of RAM |
| GPU | NVIDIA H100 NVL, `nvcc` 12.0, `sm_90` |
| `cucc` | commit 7e19fae, programs served by `cpusimd` (62 cores reserved) |
| OpenMP | clang 19, `-O3 -march=native -fopenmp`, 128 threads, `OMP_PROC_BIND=spread` |
| Run time | each program's own report of its kernels' time (the same line in the three forms), best of 2 runs |
| Correctness | each program's own verdict (PASS/FAIL), when it prints one |
| Compile time | the whole `make` of each form, one form at a time |

The script is `bench/overview/run.sh`; the raw results and the program lists are in `bench/overview/results/` ([tutorial 6](../../tutorials/06-tracking.md)). Total wall time is not used for the comparison: it includes generating and checking data on the host, the same in the three forms, which pulls the ratios toward 1 (by total time `cucc` is at 0.75x of the H100 and 1.11x of OpenMP on the first draw). HeCBench's OpenMP versions are written for GPU offload (`target teams`); on the host they run as parallel loops, and some do less work than they should, so an OpenMP time counts only when its program checks its own result or its loop is plain work sharing.

## Kernel time

`cucc`'s kernel time against the other two, per program, in bands (a ratio below 1.1 either way is a tie):

| Band | vs OpenMP (27 programs) | vs H100 (48 programs) |
|---|---|---|
| `cucc` slower, over 10x | 2 | 20 |
| `cucc` slower, 3 to 10x | 4 | 15 |
| `cucc` slower, 1.5 to 3x | 1 | 8 |
| `cucc` slower, 1.1 to 1.5x | 2 | 0 |
| tie | 3 | 2 |
| `cucc` faster, 1.1 to 1.5x | 2 | 2 |
| `cucc` faster, 1.5 to 3x | 7 | 1 |
| `cucc` faster, 3 to 10x | 5 | 0 |
| `cucc` faster, over 10x | 1 | 0 |
| **Total** | **faster 15, tie 3, slower 9** | **faster 3, tie 2, slower 43** |

Both draws gave the same picture: geomean of the H100's time over `cucc`'s 0.13 and 0.12, of OpenMP's 0.96 and 1.19.

- **Ahead of OpenMP:** hellinger 13x, hausdorff 9.8x, page-rank 5.5x, phmm 3.7x, bitonic-sort 3.5x, su3 3.4x, complex 2.9x, pool 2.8x. Kernels with much arithmetic per thread, which 16-lane vectors execute better than OpenMP's scalar loops.
- **Behind OpenMP:** memtest 175x, kalman 10x, bonds 8x, ga 3.5x, extrema 3.4x, zoom 3.1x, background-subtract 1.6x.
- **Against the H100:** `cucc` wins on sph (2.6x) and ties on colorwheel, threadfence, hmm and mixbench. On bn the GPU dumps core and `cucc` passes.

## What did not run

Of the 72, 48 have a kernel time on both `cucc` and the GPU. The others:

| Programs | Why |
|---|---|
| attention, damage, accuracy, adjacent, attentionMultiHead, bgmv, sort | CUB or Thrust (NVIDIA libraries, out of scope): do not compile on `cucc` |
| cmp, d2q9-bgk, myocyte, frna, geodesic, mriQ, mt, snake | need data files not in the repository: fail on the GPU too |
| ans, multimaterial, xlqc | do not build for the GPU on this server either (GSL and other dependencies) |
| feynman-kac | fails its own check in all three forms |
| bn | dumps core on the GPU, passes on `cucc` |
| atomicReduction, crc64, heat2d | pass, but print no kernel time (by total time `cucc` is at 0.8 to 1.4x of the GPU) |
| lrn | measured separately (it passed the 120 s limit): kernels about 12x slower than the GPU's; prints no verdict |

On OpenMP only 27 programs count: 13 versions report their own result wrong on the host (colorwheel, convolutionSeparable, damage, fft, hmm, s8n, adv, asmooth, filter, p4, threadfence, fpdc, bn), others do not build or crash, pns ends in 10 ms without doing the work, and the times mdh and rmsnorm print do not match their CUDA version's.

## Compile time

| 72 programs | `nvcc` (GPU) | `cucc` | clang OpenMP |
|---|---|---|---|
| Median per program | 1.3 s | 1.7 s | 0.4 s |
| 90% of programs within | 3.0 s | 4.4 s | 0.7 s |
| Slowest | 6.0 s | 22.4 s (frna) | 2.6 s |
| Sum over the programs that built | 116 s (69) | 170 s (62) | 30 s (67) |
| Median binary | 830 KB | 355 KB | 25 KB |

`cucc`'s binary carries a module per class or shape of block, the generic module and the bitcode for the JIT ([35](35-compile-time.md)).

## The slow cases, explained

- **Grids made for the GPU.** In memtest each thread copies its own 1 MB chunk and only 2048 of 65536 threads have work, all in the first 32 blocks. In 16 lanes every instruction is a gather or scatter over 16 different pages, and 32 of 124 workers do all the work: 4.2 s per launch, against 0.036 s on the GPU and 0.024 s for OpenMP's 2048 sequential `memcpy`.
- **Atomics on one address from every thread.** atomicIntrinsics has 2^27 threads doing 9 atomics each on the same 9 words. Every core contends for the same cache lines: 41.8 s per launch, against 0.23 s on the GPU (182x). OpenMP takes 21 s and gets the result wrong. Atomics on a uniform address are already merged per warp for add, sub, the bitwise operations, min and max ([30](../iteration-3/30-launches-and-hot-spots.md)); the kernel also uses increment and decrement with wrap, exchange and compare-and-swap, which are not, and even one atomic per warp leaves 124 workers contending for 9 lines that a GPU serves from its L2.
- **Single-precision transcendentals computed in double.** libmvec's vector `expf` errs by up to about 4 ulp, above CUDA's bound; the double version rounded to float stays under 1 ulp ([24](../iteration-3/24-16-bit-types.md)). bilateral calls `expf` 49 times per pixel and runs 73x behind the H100: each call is an `exp` on 8 doubles. A float implementation within 1 to 2 ulp would recover most of it.
- **Short kernels.** A launch through the daemon costs 5 to 7 us in the socket round trip alone; lif's kernel takes 4 us on the GPU and 100 us here.

Back to the [iteration 4 index](README.md).
