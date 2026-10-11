# CUDA samples

Reference CUDA programs for CudaSimulator. Each one runs on the real GPU, compares the result with a CPU implementation and prints `PASS`/`FAIL`. Afterwards, the same sources serve as input to our compiler: the output must remain `PASS`.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Device code is compiled with `-fmad=false` (option `SAMPLES_NO_FMAD`), so that float results are bit-for-bit equal to those of the CPU wherever the math is exact IEEE.

## Workloads

The workloads were sized so that each kernel lasts at least 100 ms on the RTX 5060 Laptop when that is possible: with kernels of ~1 ms, what is measured in the simulator is the launch and communication overhead, not the generated code. Memory-bandwidth-bound kernels do not reach 100 ms within the board's 8 GB (they would need ~30 GB of traffic); for those, the largest reasonable size was used. The CPU references are naive and single-threaded, so some samples take 1 to 2 minutes.

| Sample | Workload | Time on the RTX 5060 | Bound by |
|---|---|---|---|
| `matmul` | 4100 x 4130 x 4070 | naive 169 ms, tiled 129 ms | compute |
| `matmul_naive` | same as `matmul` | 169 ms | compute |
| `nbody` | 175,000 bodies, 1 step | 104 ms | compute |
| `mandelbrot` | 4096 x 3072, up to 22,528 iterations | 102 ms | compute (divergent) |
| `bitonic_sort` | 240,000 segments of 1024 | 103 ms | shared memory + barriers |
| `histogram` | 671 M bytes | global 111 ms, shared 6 ms | atomics (global) / memory |
| `heat2d` | 8000 x 6000, 60 steps | 102 ms in total (1.7 ms per launch) | memory |
| `scan` | 512 M integers | 34 ms (3 levels) | memory |
| `spmv` | 8 M rows, up to 64 nnz per row | scalar 33 ms, warp 19 ms | memory |
| `transpose` | 24,000 x 18,000 | naive 15 ms, tiled 11 ms | memory |
| `reduction` | 268 M floats | 3.3 ms | memory |
| `tensor_gemm` | 4096 x 4096 x 4096 with WMMA (bf16, fp16, int8) | 8.5 to 9.8 ms per type (14 to 16 TFLOPS) | tensor cores |

## What each sample exercises

| Sample | Shared mem | `__syncthreads` | Divergence | Memory | Atomics | Warp intrinsics |
|---|---|---|---|---|---|---|
| `matmul` | 16x16 tiles | inside loop | early return (naive) | contiguous + strided | | |
| `nbody` | `float4` tile, uniform read (broadcast) | inside loop | tail of the last block | `float4` | | |
| `reduction` | tree | inside loop | shrinking active set | grid-stride | float, uniform address | `__shfl_down_sync` |
| `histogram` | 256 bins | init / merge | conditional merge | byte load | int, heavy conflicts, shared and global | |
| `heat2d` | tile + halo | 1 per launch | halo only at block edges, boundary condition | clamp, 2D partial blocks | | |
| `scan` | Blelloch | 1 per level (18) | active set halves and doubles back | indices computed in shared | | |
| `transpose` | tile with `[32][33]` padding | 1 | edges | strided write (scatter) vs contiguous | | |
| `bitonic_sort` | segment of 1024 | nested loop (55 per block) | swap depends on the data | bitwise indices | | |
| `spmv` | | | per-lane trip count (0 to 64) | real gather `x[col[k]]` | | `__shfl_down_sync` |
| `mandelbrot` | | | divergent loop, mask shrinks lane by lane | | | |

## Runtime API used

Only the minimum, which is the first surface our runtime will need to implement:

`cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemset`, `cudaDeviceSynchronize`, `cudaGetLastError`, `cudaGetErrorString`

Language features used: `__global__`, `__device__`, `__host__ __device__`, static `__shared__`, `threadIdx/blockIdx/blockDim/gridDim` in 1D and 2D, `dim3`, `float3/float4`, `min/max`, `rsqrtf`, `atomicAdd` (int and float), `__shfl_down_sync`, `constexpr` used on the device.
